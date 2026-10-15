
from utils.gui_funcs import select_directory
from utils.analyze_gratings import main as gratings

def main():

    gt_dir = select_directory('Select gratings directory')
    gratings(gt_dir)

    