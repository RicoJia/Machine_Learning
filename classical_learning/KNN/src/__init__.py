from sklearn.metrics import accuracy_score as accuracy

from .collaborative_filtering import collaborative_filtering
from .distances import cosine_distances, euclidean_distances, manhattan_distances
from .k_nearest_neighbor import KNearestNeighbor
from .load_json_data import load_json_data
from .load_movielens import load_movielens_data
