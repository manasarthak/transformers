import datasets
import transformers
from transformers.convert_slow_tokenizer import SLOW_TO_FAST_CONVERTERS
from transformers.utils import logging
from collections import Counter
import argparse
from concurrent.futures import ThreadPoolExecutor

logging.set_verbosity_info()

# Map tokenizer classes
TOKENIZER_CLASSES = {
    name: (getattr(transformers, name), getattr(transformers, name + "Fast"))
    for name in SLOW_TO_FAST_CONVERTERS
}

def load_dataset():
    return datasets.load_dataset("facebook/xnli", split="test+validation")

def check_diff(spm_diff, tok_diff, slow, fast):
    """
    Compare token differences and try to resolve conflicts between slow and fast tokenizers.
    """
    if spm_diff == list(reversed(tok_diff)) or (len(spm_diff) == len(tok_diff) and fast.decode(spm_diff) == fast.decode(tok_diff)):
        return True

    spm_reencoded = slow.encode(slow.decode(spm_diff))
    tok_reencoded = fast.encode(fast.decode(spm_diff))

    if spm_reencoded != spm_diff and spm_reencoded == tok_reencoded:
        return True
    
    return False

def check_details(line, spm_ids, tok_ids, slow, fast):
    """
    Analyze tokenization differences in more detail.
    """
    first, last = 0, len(spm_ids)

    # Detect the first difference between slow and fast tokenizer
    for i, (spm_id, tok_id) in enumerate(zip(spm_ids, tok_ids)):
        if spm_id != tok_id:
            first = i
            break

    # Detect the last difference
    for i, (spm_id, tok_id) in enumerate(zip(reversed(spm_ids), reversed(tok_ids))):
        if spm_id != tok_id:
            last = len(spm_ids) - i
            break

    spm_diff = spm_ids[first:last]
    tok_diff = tok_ids[first:last]

    # Perform detailed comparison
    if check_diff(spm_diff, tok_diff, slow, fast):
        return True

    return False

def test_string(slow, fast, text):
    """
    Test tokenization consistency between slow and fast tokenizer.
    """
    global perfect, imperfect, wrong, total
    slow_ids = slow.encode(text)
    fast_ids = fast.encode(text)

    total += 1

    if slow_ids != fast_ids:
        if check_details(text, slow_ids, fast_ids, slow, fast):
            imperfect += 1
        else:
            wrong += 1
    else:
        perfect += 1

    if total % 10000 == 0:
        print(f"Processed: {total}, Perfect: {perfect}, Imperfect: {imperfect}, Wrong: {wrong}")

def test_tokenizer(slow, fast, dataset):
    """
    Run tokenizer tests on the dataset.
    """
    for i in range(len(dataset)):
        # Test premises for all languages
        for text in dataset[i]["premise"].values():
            test_string(slow, fast, text)

        # Test hypotheses for all languages
        for text in dataset[i]["hypothesis"]["translation"]:
            test_string(slow, fast, text)

def main(checkpoint=None):
    """
    Main function to test tokenizers.
    """
    global perfect, imperfect, wrong, total
    dataset = load_dataset()

    for name, (slow_class, fast_class) in TOKENIZER_CLASSES.items():
        checkpoint_names = list(slow_class.max_model_input_sizes.keys()) if checkpoint is None else [checkpoint]

        for ckpt in checkpoint_names:
            perfect = 0
            imperfect = 0
            wrong = 0
            total = 0

            print(f"===== Testing {name}: {ckpt} =====")
            slow = slow_class.from_pretrained(ckpt)
            fast = fast_class.from_pretrained(ckpt)

            # Parallel execution for faster processing
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(test_tokenizer, slow, fast, dataset)]
                for future in futures:
                    future.result()  # Ensure all tests complete

            print(f"Results for {ckpt}: Perfect: {perfect}, Imperfect: {imperfect}, Wrong: {wrong}, Total: {total}")
            print(f"Accuracy: {perfect * 100 / total:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test slow and fast tokenizers.")
    parser.add_argument('--checkpoint', type=str, default=None, help="Specific checkpoint to test")
    args = parser.parse_args()

    main(checkpoint=args.checkpoint)
