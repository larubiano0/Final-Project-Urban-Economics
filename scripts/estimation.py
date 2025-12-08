import pandas as pd
import os 

sim_chat_panel_full_path = os.path.join("sim_chat_panel_full.csv")
sim_chat_panel_full = pd.read_csv(sim_chat_panel_full_path)

print(sim_chat_panel_full.head(10))