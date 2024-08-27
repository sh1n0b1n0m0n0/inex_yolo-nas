cd /home/alexsh/darknet_experiments/src/Object-Detection-Metrics && \
python pascalvoc.py \
    -det /home/alexsh/yolo-nas/runs/exp4/pascal_voc_det \
    -gt /home/alexsh/yolo-nas/runs/exp4/pascal_voc_gt \
    -sp /home/alexsh/yolo-nas/runs/exp4/ODM_results\
    --noplot

cd /home/alexsh/yolo-nas