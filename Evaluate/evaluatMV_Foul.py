from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from SoccerNet.Evaluation.MV_FoulRecognition import evaluate

if __name__ == '__main__':
    parser = ArgumentParser(description='my method', formatter_class=ArgumentDefaultsHelpFormatter)    
    parser.add_argument('--gs_file',   required=True, type=str, help='Path to the json groundtruth' )
    parser.add_argument('--prediction_file',   required=True, type=str, help='Path to the json prediction' )
    args = parser.parse_args()


    results = evaluate(args.gs_file, args.prediction_file)
    print(results)


    
