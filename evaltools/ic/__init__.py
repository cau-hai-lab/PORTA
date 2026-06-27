import os
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

def coco_caption_eval(annotation_file, results_file):
    # 디코더에서 생성된 결과 파일을 평가하기 위한 함수
    print("[Debug] evaltools/ic/__init__.py -> coco_caption_eval()함수 호출 : generate 결과 평가")
    assert os.path.exists(annotation_file)

    # create coco object and coco_result object
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # evaluate results
    coco_eval.evaluate() #평가 객체 반환

    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f'{metric}: {score:.3f}', flush=True)

    return coco_eval
