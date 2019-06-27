引き継ぎ　プログラム仕様書（2017/3月 石伏　智)

場所の名前、機能記述情報を入力するGUIのプログラムです

使用コマンド
$python Place_data_generator.py "データセット" ”clip_num"

clip_numはGUIに地図を表示する時に地図の大きさを指定するための値です


Emergent_system_lab_high_cameraの場合は”em”としてください
kitano_labの場合は"kitano"としてください
wada_labの場合は"wada"としてください
tsubo_labの場合は"tsubo"としてください

使用例
emlabの場合

$ python Place_data_generator.py training_dataset/Emergent_system_lab_high_camera/ em

tsubolabの場合
$python Place_data_generator.py training_dataset/tsubo_lab/ tsubo




