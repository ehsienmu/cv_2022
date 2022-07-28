* 如果要跑baseline model的話，已在default設定好，直接輸入
python3 main.py即可開始訓練模型，最後訓練出的model .pt file會在./save_dir/pretrained_resnet50/best_model.pt

* evaluate時也可直接輸入
python3 eval.py

* main.py
--model預設為pretrained_resnet50

* eval.py的
--model預設為pretrained_resnet50
--path預設為./save_dir/pretrained_resnet50/best_model.pt