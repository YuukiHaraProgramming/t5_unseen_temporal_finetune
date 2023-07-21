import argparse
import torch
from src.model import T5FineTuner
from datasets import load_dataset
from tqdm import tqdm

class T5FineTunerEvaluation:
    def __init__(self, model_name, checkpoint_path):

        self.finetuned_pred = T5FineTuner.load_from_checkpoint(
            checkpoint_path=checkpoint_path, model_name=model_name)

        self.model = self.finetuned_pred.model
        self.tokenizer = self.finetuned_pred.tokenizer

        self.model.eval()

    def evaluate(self, test_dataset_path):
        test_dataset = load_dataset('json', data_files={'test':test_dataset_path}, field='data')['test']

        f1_list = []
        with torch.no_grad():
            for i, (year, src, target) in enumerate(zip(tqdm(test_dataset['year']), test_dataset['src'], test_dataset['tgt'])):
                prompt = f'year: {year} text: {src}'
                input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)

                outputs = self.model.generate(input_ids=input_ids)
                decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                if len(decoded_output.split()) == 0:
                    print(f'''
                          input: {prompt} \n
                          ouptput: {decoded_output} \n
                          target: {target}
                          ''')
                    max_f1 = 0
                else:
                    max_f1 = max([self.calc_f1_score(decoded_output, t) for t in target])

                # print(f'''
                #       prompt: {prompt} \n
                #       output: {decoded_output} \n
                #       target: {target} \n
                #       max_f1: {max_f1} \n
                #       ============================
                #       ''')

                f1_list.append(max_f1)

            print(f'averaged F1 score: {sum(f1_list)/len(f1_list)}')

    def calc_f1_score(self, output, target):
        output_ent = output.split()
        target_ent = target.split()
        n = len(set(output_ent) & set(target_ent))

        precision = n / len(output_ent)
        recall = n / len(target_ent)
        f1 = (2 * precision * recall) / (precision + recall) if n != 0 else 0
        return f1



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_file", required=True)
    parser.add_argument("--model_name", default='t5-small')
    parser.add_argument("--test_dataset_path", required=True)
    args = parser.parse_args()

    finetuned_model = T5FineTunerEvaluation(model_name=args.model_name, checkpoint_path=args.checkpoint_file)
    finetuned_model.evaluate(args.test_dataset_path)
