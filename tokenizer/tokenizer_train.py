import sentencepiece as spm

input = '../data.txt'
output = 'hu'

spm.SentencePieceTrainer.Train('--input=' + input +
                               ' --model_prefix=' + output +
                               ' --character_coverage=1.0' +
                               ' --vocab_size=' + "512" +
                               ' --model_type=bpe' +
                               ' --max_sentence_length=100000' +
                               ' --split_by_whitespace=true')

