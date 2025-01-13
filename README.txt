Link Google drive: https://drive.google.com/drive/folders/1Y6iKM4XqUU_3QkvJspw_2cN_tyZmKMg4?usp=drive_link
Part 2: Pong game
Link video: https://drive.google.com/file/d/1WS0WgzcCxLTWm9IYho_7t1Iu3tZnJD05/view?usp=drive_link
link model: https://drive.google.com/file/d/115Av2Uj9SxFsj6h4n-J46Tpgcjz3hMlq/view?usp=drive_link
Test the model and record a short video
Name all your source code and video files using the following template: pong_model_123456789. (model .pkl) 
Training hours: 2 hours and 20 minutes
Command lines were used:
AIClass\Scripts\activate
Python train_pong.py
python -m baselines.run --alg=deepq --env=PongNoFrameskip-v4 --load_path=C:\Users\dminh\Minh-wkspc\ITCS5153-AI\lab5\baselines\baselines\deepq\experiments\pong_model_801131341.pkl --num_timesteps=0 --play

Part 3: Other pong game - Road Runner game
Link video: https://drive.google.com/file/d/1GqY3cPDo2bZ9-k8Io7CP2nQId-k7i8ov/view?usp=drive_link
link model: https://drive.google.com/file/d/1dgaGUyrEGZ8_Kpbpd36F3vltL4I6LR_U/view?usp=drive_link
The command lines were used:
AIClass\Scripts\activate
Python train_roadRunner.py
python -m baselines.run --alg=deepq --env=RoadRunnerNoFrameskip-v4 --load_path=C:\Users\dminh\Minh-wkspc\ITCS5153-AI\lab5\baselines\baselines\deepq\experiments\RoadRunner_model_801131341.pkl --num_timesteps=0 --play
