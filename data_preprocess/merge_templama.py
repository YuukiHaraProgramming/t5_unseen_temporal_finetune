'''
Concat the templama datasets (train.json, val.json, test.json).
'''
import os
import json
from argparse import ArgumentParser

'''
format

training data entry:
        {"year": "2010"
        "src": "Valentino Rossi plays for <extra_id_0>.",
        "tgt": [
            "Yamaha Motor Racing"
        ]}

'''

def merge_json_files(data_dir, out_dir):
        # Create the all.json file structure
    all_data = {"name": "templama_train_and_test", "data": []}

    # Iterate over the train, val, and test JSON files
    for filename in ['train', 'test']:
        file_path = os.path.join(data_dir, f'{filename}.json')

        # Read the JSON file
        with open(file_path, 'r') as file:
            json_data = json.load(file)

        # Merge the data into all_data
        all_data['data'].extend(json_data['data'])

    # Write the merged data to all.json
    all_file_path = os.path.join(out_dir, 'train_and_test.json')
    with open(all_file_path, 'w') as file:
        json.dump(all_data, file, indent=4)

    print("Merged JSON files successfully.")

def main():
    parser = ArgumentParser()
    parser.add_argument('--data_dir', default='/mnt/mint/hara/datasets/templama/preprocessed/multiple_answers_list')
    parser.add_argument('--out_dir', default='/mnt/mint/hara/datasets/templama/preprocessed/selective')
    args = parser.parse_args()

    merge_json_files(args.data_dir, args.out_dir)

if __name__ == '__main__':
    main()