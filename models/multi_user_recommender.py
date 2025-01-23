import torch
import os
from constants import DATA_PATH

embedding_manager = torch.load(os.path.join(DATA_PATH, 'embedding_manager.pt'))

embedding_manager.get_top_movies_for_user("danicrg", 20)
embedding_manager.get_top_movies_for_user("danicrg", 20)
