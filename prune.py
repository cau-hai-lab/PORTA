import os
import time
import datetime
import torch
import pandas as pd
import ruamel.yaml as yaml
import lightning as L
from argparse import ArgumentParser

from pruners import available_pruners, get_pruner_by_name
from datasets import create_dataset, create_loader, create_sampler

from utils.misc import millions
from utils.prune_utils import save_prunable_model, named_masked_parameters
from utils.model_utils import available_models, model_factory
from utils.functions import get_unprunable_parameters


def cuda_sync_if_available():
    """Synchronize CUDA kernels before/after wall-clock timing."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _debug_excludes(model, keys):
    matched = []
    for n, _ in model.named_parameters():
        if any(k in n for k in keys):
            matched.append(n)
    print(f"[Debug] 제외 매칭 파라미터 수: {len(matched)}")
    for x in matched[:30]:
        print("  -", x)


def _strip_wrappers(name):
    while name.startswith("module."):
        name = name[len("module."):]
    if name.startswith("_forward_module."):
        name = name[len("_forward_module."):]
    return name


def _scope_key_for_name(pname, model_name=None):
    name = _strip_wrappers(pname)
    bare = name[len("model."):] if name.startswith("model.") else name

    if model_name == "llava":
        if bare.startswith("language_model."):
            return "text"
        if bare.startswith("vision_tower."):
            return "vision"
        if bare.startswith("multi_modal_projector."):
            return "fusion"

    if model_name == "qwen_vl":
        if bare.startswith("visual.merger."):
            return "fusion"
        if bare.startswith("visual."):
            return "vision"
        if bare.startswith("layers.") or bare.startswith("embed_tokens.") or bare.startswith("lm_head."):
            return "text"

    if bare.startswith("language_model."):
        return "text"
    if bare.startswith("model.layers.") or bare.startswith("model.embed_tokens.") or bare.startswith("layers.") or bare.startswith("embed_tokens."):
        return "text"
    if bare.startswith("vision_tower.") or bare.startswith("vision_model.") or bare.startswith("visual."):
        return "vision"
    if bare.startswith("multi_modal_projector."):
        return "fusion"
    return "other"


def summarize_pruning_scope(model, exclude=(), show_samples=5, include_bias=False, model_name=None):
    """
    Summarize currently masked parameters by modality scope.
    """
    buckets = ("text", "vision", "fusion", "other")
    counts = {k: 0 for k in buckets}
    names = {k: [] for k in buckets}
    totals = {k: 0 for k in buckets}
    pruned = {k: 0 for k in buckets}

    for pname, mask, param in named_masked_parameters(model, bias=include_bias, exclude=exclude):
        key = _scope_key_for_name(pname, model_name=model_name)

        counts[key] += 1
        totals[key] += mask.numel()
        pruned[key] += mask.numel() - mask.sum().item()

        if len(names[key]) < show_samples:
            names[key].append(pname)

    print("\n[Pruning Scope Summary]")
    for k in buckets:
        rate = 0.0 if totals[k] == 0 else (pruned[k] / totals[k] * 100.0)
        print(
            f"- {k:6s}: params_with_masks={counts[k]:4d}, "
            f"mask_elems={totals[k]:8d}, pruned_elems={int(pruned[k]):8d} ({rate:5.2f}%)"
        )
        for n in names[k]:
            print(f"    · {n}")
    print()

    return counts, totals, pruned


def main(args):
    job_wall_start = time.perf_counter()
    setup_wall_start = job_wall_start

    # Pruning is done with fp32 precision.
    torch.set_float32_matmul_precision("high")

    fabric = L.Fabric(
        accelerator="cuda",
        devices=1,
        precision="32-true",
    )

    # Reproducibility.
    fabric.seed_everything(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    print("<사전 학습된 모델 가져오기>")
    model = model_factory(model_name=args.model)
    print("<사전 학습된 모델 가져옴>")

    init_args = [model]

    keys = get_unprunable_parameters(args.model, prune_scope=args.prune_scope)
    init_kwargs = {"keys_to_exclude": keys}
    _debug_excludes(model, keys)

    general_dataset = None
    region_dataset = None
    general_loader = None
    region_loader = None
    config_for_dataset = None

    # Some pruning algorithms do not need data.
    if args.pruner in ("rand", "omp", "lamp", "l2"):
        score_args = {}
        score_kwargs = {}

    # Data-dependent pruners.
    elif args.pruner in (
        "snip",
        "itersnip",
        "chita",
        "multiflow",
        "tamt",
        "smoothflow",
        "ecoflap",
        "dlp",
        "varmin",
        "wanda",
        "sparsegpt",
        "wandavlm",
        "smoothflow_dakdak",
        "varmin_col",
        "smoothflow_dakdak_col",
        "smoothflow_dakdak_plot",
        "wanda_multi",
        "sparsegpt_multi",
        "wanda_ours",
        "sparsegpt_ours",
        "smoothflow_dakdak_eco",
        "smoothflow_dakdak_grad",
        "smoothflow_dakdak_zeromean",
        "smoothflow_dakdak_diag",
        "wanda_eco",
        "wanda_etap",
        "smoothflow_dakdak_multi",
        "varmin_range",
        "porta",
    ):
        dataset_config_path = args.dataset_config or f"configs/prune/general_loader_{args.model}.yaml"
        print(f"Loading dataset config from {dataset_config_path}.")
        print("<Pruning에 필요한 dataset 불러오기 -> datasets/pretrain_dataset.py로 이동>")

        config_for_dataset = yaml.load(open(dataset_config_path, "r"), Loader=yaml.Loader)

        if args.model == "flamingo":
            config_for_dataset["tokenizer_obj"] = getattr(model, "tokenizer", None)
            config_for_dataset["image_processor_obj"] = getattr(model, "image_processor", None)

            if config_for_dataset["tokenizer_obj"] is None:
                raise ValueError("Flamingo tokenizer is missing on model. Did you set model.tokenizer?")

        general_dataset, region_dataset = create_dataset(
            f"pretrain_{args.model}",
            config=config_for_dataset,
        )

        general_sampler, region_sampler = create_sampler(
            datasets=[general_dataset, region_dataset],
            shuffles=[True, True],
            num_replicas=fabric.world_size,
            global_rank=fabric.global_rank,
            is_eval=[True, True],
        )

        [general_loader, region_loader] = create_loader(
            [general_dataset, region_dataset],
            samplers=[general_sampler, region_sampler],
            batch_size=[config_for_dataset["batch_size"], config_for_dataset["batch_size"]],
            num_workers=[8, 8],
            is_trains=[False, False],
            collate_fns=[
                getattr(general_dataset, "collate_fn", None),
                getattr(region_dataset, "collate_fn", None),
            ],
        )

        general_loader, region_loader = fabric.setup_dataloaders(
            general_loader,
            region_loader,
            use_distributed_sampler=False,
        )
        model = fabric.setup_module(model)

        from math import ceil

        bsz = config_for_dataset["batch_size"]
        if args.samples is not None:
            args.num_batches = max(1, ceil(args.samples / bsz))
            print(
                f"[Calib] samples={args.samples}, batch_size={bsz} -> "
                f"num_batches_per_step={args.num_batches}"
            )

        config_for_dataset.update(vars(args))

        score_args = [model]
        score_kwargs = {
            "dataloader": general_loader,
            "region_loader": region_loader if model.name != "dino" else None,
            "device": fabric.device,
            "config": config_for_dataset,
            "fabric": fabric,
            "num_batches_per_step": args.num_batches,
            "pruning_steps": args.epochs,
            "schedule": args.schedule,
            "lambda_": args.lambda_,
            "alpha": args.alpha,
            "model_name": args.model,
        }

        print(f"<general_dataset 크기: {len(general_dataset)}>")
        print(f"<region_dataset 크기: {len(region_dataset)}>")

        if len(general_dataset) == 0 or len(region_dataset) == 0:
            print("<데이터셋이 비어 있음>")
            exit()

        print("<데이터셋 로딩 성공>")
        print(f"<general_loader batch_size: {config_for_dataset['batch_size']}>")
        print(f"<region_loader batch_size: {config_for_dataset['batch_size']}>")

    else:
        raise ValueError(f"Unsupported pruner: {args.pruner}")

    print("<Pruner 객체 생성>")
    pruner_init_start = time.perf_counter()
    pruner = get_pruner_by_name(args.pruner, *init_args, **init_kwargs)
    cuda_sync_if_available()
    pruner_init_s = time.perf_counter() - pruner_init_start

    setup_s = time.perf_counter() - setup_wall_start

    summarize_pruning_scope(pruner.model, exclude=keys, model_name=args.model)

    if hasattr(pruner, "requires_training") and pruner.requires_training:
        pruner.model.train()
    else:
        pruner.model.eval()

    runtimes = {"sparsity": [], "runtime": []}
    timing_rows = []

    for sparsity_string in args.sparsities.split(","):
        sparsity = float(sparsity_string) / 100.0

        print("<Pruning 시작, prune.py -> pruner.prune() 실행>")
        score_kwargs["output_dir"] = args.output_dir

        # ------------------------------------------------------------------
        # Official pruning time:
        #   pruner.prune() 시작부터 종료까지.
        #
        # Includes:
        #   - calibration forward
        #   - activation/statistic collection
        #   - score computation
        #   - sparsity allocation
        #   - mask generation
        #
        # Excludes:
        #   - model loading
        #   - dataset/dataloader construction
        #   - pruner initialization
        #   - mask saving
        #   - downstream evaluation
        # ------------------------------------------------------------------
        cuda_sync_if_available()
        time_start = time.perf_counter()
        prune_wall_start = time.perf_counter()

        pruner.prune(sparsity, *score_args, **score_kwargs)

        cuda_sync_if_available()
        prune_wall_s = time.perf_counter() - prune_wall_start
        time_end = time.perf_counter()

        summarize_pruning_scope(pruner.model, exclude=keys, model_name=args.model)

        runtimes["runtime"].append(time_end - time_start)
        runtimes["sparsity"].append(sparsity_string)

        # ------------------------------------------------------------------
        # Save mask/weights separately. This is excluded from pruning time.
        # ------------------------------------------------------------------
        last_folder = args.output_dir.split("/")[-1]
        if last_folder != str(pruner):
            args.output_dir = os.path.join(args.output_dir, str(pruner))
            os.makedirs(args.output_dir, exist_ok=True)

        pruned_model_path = os.path.join(
            args.output_dir,
            f"{args.model}_{pruner}_{sparsity_string}_seed{args.seed}.pth",
        )

        cuda_sync_if_available()
        save_wall_start = time.perf_counter()

        params_path, mask_path = save_prunable_model(
            model,
            pruned_model_path,
            mask_only=not pruner.modifies_weights,
        )

        save_wall_s = time.perf_counter() - save_wall_start

        print(f"Saved mask at {mask_path}")
        if pruner.modifies_weights:
            print(f"Saved params at {params_path}")

        remaining_params, total_params = pruner.stats()
        print(
            f"Sparsity: {sparsity_string}%",
            f"Remaining params (M): {millions(remaining_params, decimals=2)}",
            f"Total params (M): {millions(total_params, decimals=2)}",
            f"Remaining: {remaining_params / total_params * 100:.2f}%\n\n",
            sep="\n",
            end="\n\n",
        )

        total_wall_s = time.perf_counter() - job_wall_start
        pruner_reported_s = getattr(pruner, "scoring_time", None)

        timing_row = {
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "model": args.model,
            "pruner": str(pruner),
            "requested_pruner": args.pruner,
            "sparsity": sparsity_string,
            "seed": args.seed,
            "samples": args.samples,
            "num_batches_per_step": args.num_batches,
            "alpha": args.alpha,
            "prune_scope": args.prune_scope,
            "dataset_config": args.dataset_config or f"configs/prune/general_loader_{args.model}.yaml",

            # Setup / init.
            "setup_wall_s": setup_s,
            "pruner_init_wall_s": pruner_init_s,

            # Main pruning-time number.
            "prune_wall_s": prune_wall_s,

            # Optional internal value from each pruner.
            # Use this only as a sanity check because its definition can vary.
            "pruner_reported_scoring_s": pruner_reported_s,

            # Excluded from pruning time.
            "save_wall_s": save_wall_s,
            "total_wall_s": total_wall_s,

            # Sparsity stats.
            "remaining_params": int(remaining_params),
            "total_params": int(total_params),
            "remaining_pct": float(remaining_params / total_params * 100.0),

            # Paths.
            "mask_path": mask_path,
            "output_dir": args.output_dir,
        }
        timing_rows.append(timing_row)

        print(
            "[Timing] "
            f"setup={setup_s:.2f}s, "
            f"pruner_init={pruner_init_s:.2f}s, "
            f"prune_wall={prune_wall_s:.2f}s, "
            f"pruner_reported={pruner_reported_s}, "
            f"save={save_wall_s:.2f}s, "
            f"total={total_wall_s:.2f}s"
        )

        if hasattr(pruner, "reset"):
            try:
                pruner.reset()
            except Exception as exc:
                print(f"[WARN] pruner.reset() failed after mask save: {type(exc).__name__}: {exc}")
        else:
            print(f"[WARN] {type(pruner).__name__} has no reset(); continuing after mask save")

    if pruner.is_one_shot:
        scores_path = os.path.join(args.output_dir, f"{args.model}_{args.pruner}_scores.pth")
        try:
            torch.save(pruner.state_dict(), scores_path)
            print(f"Saved scores at {scores_path}. Finished!")
        except Exception as exc:
            print(f"[WARN] could not save pruner scores after mask save: {type(exc).__name__}: {exc}")

    runtimes = pd.DataFrame(runtimes)
    runtimes.to_csv(
        os.path.join(args.output_dir, f"{args.model}_{args.pruner}_seed{args.seed}_runtimes.csv"),
        index=False,
    )

    if timing_rows:
        timing_df = pd.DataFrame(timing_rows)
        timing_csv = args.timing_csv or os.path.join(
            args.output_dir,
            f"{args.model}_{args.pruner}_seed{args.seed}_timing_detailed.csv",
        )
        os.makedirs(os.path.dirname(timing_csv) or ".", exist_ok=True)
        timing_df.to_csv(timing_csv, index=False)
        print(f"Saved detailed timing at {timing_csv}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-p", "--pruner", type=str, required=True, choices=available_pruners)
    parser.add_argument("-m", "--model", type=str, required=True, choices=available_models)
    parser.add_argument(
        "-s",
        "--sparsities",
        type=str,
        default="63,75,90",
        help="comma separated list of sparsities to prune at. Default: 63,75,90",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for the random number generator. Default: 42")

    parser.add_argument(
        "--num_batches",
        default=3000,
        type=int,
        help=(
            "number of batches to use. "
            "If epochs > 1, then these will be the batches used at each pruning iteration. "
            "If epochs == 1, then these will be the total batches processed. Default: 3000."
        ),
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=1,
        help=(
            "the total number of pruning iterations. "
            "This argument is only used by pruners relying on iterations, so IterSNIP and CHITA++. "
            "If you select the pruner 'chita' and provide this value greater than 1, it will directly run CHITA++. "
            "Default: 1"
        ),
    )
    parser.add_argument(
        "--schedule",
        type=str,
        default="exp",
        choices=["linear", "exp", "const"],
        help="schedule for IterSNIP/CHITA++. Default: exp",
    )
    parser.add_argument(
        "--output_dir",
        default="pruned_weights",
        help="directory where to dump the pruned weights. Default: ./pruned_weights",
    )
    parser.add_argument(
        "--dataset_config",
        default=None,
        help="optional calibration dataset config path. Defaults to configs/prune/general_loader_{model}.yaml",
    )
    parser.add_argument(
        "--lambda_",
        type=float,
        default=1e-5,
        help="ridge penalty for CHITA and CHITA++, unused otherwise. Default: 1e-5",
    )
    parser.add_argument(
        "--prune_scope",
        choices=["both", "text", "vision"],
        default="both",
        help="프루닝 범위: both(기존과 동일), text(텍스트 타워만), vision(비전 타워만)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1,
        help="method-specific alpha hyperparameter",
    )
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument(
        "--timing_csv",
        default=None,
        help="optional path for detailed timing CSV. Defaults to output_dir/<model>_<pruner>_seed<seed>_timing_detailed.csv",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    main(args)