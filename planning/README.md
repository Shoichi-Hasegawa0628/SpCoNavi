# Planning codes of SpCoNavi 

【Folder】  
 - `/lang_m/`: Including the word disctionary files (Japanese syllables) in Julius


【Files】  
 - `Astar_Database.py`: A star algorithm (goal position selection from database for training data in spatial concept learning)
 - `Astar_SpCp.py`: A star algorithm (goal position selection from learned spatial concepts by conventional method
 - `JuliusNbest_dec.py`: Sub-program for the N-best speech recognition by Julius
 - `README.md`: Read me file (This file)
 - `SpCoNavi0.1.py`: Main path-planning code of SpCoNavi
 - `SpCoNavi0.1_SIGVerse.py`: Main path-planning code of SpCoNavi in SIGVerse
 - `SpCoNavi0.1s.py`: Main path-planning code of SpCoNavi ()
 - `__init__.py`: Code for initial setting (PATH and parameters)
 - `__init__JSAI2019.py`: Initial setting code (backup for our experiment in real robot environment)
 - `__init__SIGVerse.py`: Initial setting code (backup for our experiment in SIGVerse)
 - `costmap.py`: Program to get costmap
 - `costmap_SIGVerse.py`: Program to get costmap for SIGVerse
 - `initJSAI2019.py`: Initial setting code (another backup for our experiment in real robot data)
 - `path_visualizer.py`: Program for visualization of path trajectory
 - `path_weight_visualizer.py`: Program for visualization of path trajectory and emission probability (log scale)
 - `path_weight_visualizer_step.py`: Program for visualization of path trajectory and emission probability (log scale)
 - `path_weight_visualizer_step_CoRL.py`: Program for visualization of path trajectory and emission probability (log scale) for each step in real robot environment
 - `path_weight_visualizer_step_SIGVerse.py`: Program for visualization of path trajectory and emission probability (log scale) for each step in SIGVerse
 - `submodules.py`: Sub-program for functions
 - `weight_visualizer.py`: Program for visualization of emission probability (log scale)

