from preprocess.parse_mat import ParseMAt
import config
import argparse


parse_mat_to_np = ParseMAt(
    overwrite=False
)
parse_mat_to_np.generate_data_all()
