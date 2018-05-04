import subprocess


AVAILABLE_ARCHS = ('lstm_wiseman_iwslt_de_en', 'lstm', 'lstm_luong_wmt_en_de',
                   'fconv_iwslt_de_en', 'fconv', 'fconv_wmt_en_ro',
                   'fconv_wmt_en_de', 'fconv_wmt_en_fr')


class Fairseq:
    def __init__(self, fairseq_path):
        self.fairseq_path = fairseq_path

    @classmethod
    def run(cls, args):
        return subprocess.run(args, stdout=subprocess.PIPE)

    def preprocess(self, data_path, data_bin_path, source_lang, target_lang,
                   source_threshold=1, target_threshold=1):
        args = ['python', '{}/preprocess.py'.format(self.fairseq_path),
                '--source-lang', source_lang,
                '--target-lang', target_lang,
                '--trainpref', '{}/train'.format(data_path),
                '--validpref', '{}/valid'.format(data_path),
                '--testpref', '{}/test'.format(data_path),
                '--destdir', data_bin_path,
                '--thresholdsrc', str(source_threshold),
                '--thresholdtgt', str(target_threshold)]
        self.run(args)

    def train(self, data_bin_path, model_path, arch='fconv_iwslt_de_en',
              learning_rate=0.25, clip_norm=0.1, dropout=0.2, max_tokens=6000):
        args = ['python', '{}/train.py'.format(self.fairseq_path),
                data_bin_path,
                '--lr', str(learning_rate),
                '--clip-norm', str(clip_norm),
                '--dropout', str(dropout),
                '--arch', arch,
                '--max-tokens', str(max_tokens),
                '--skip-invalid-size-inputs-valid-test',
                '--save-dir', model_path]
        return self.run(args).stdout.decode('utf-8').split('\n')

    def predict(self, data_bin_path, model_path, batch_size=128, beam=5):
        args = ['python', '{}/generate.py'.format(self.fairseq_path),
                data_bin_path,
                '--path', '{}/checkpoint_best.pt'.format(model_path),
                '--batch-size', str(batch_size), '--beam', str(beam),
                '--skip-invalid-size-inputs-valid-test']
        lines = self.run(args).stdout.decode('utf-8').split('\n')
        return [line for line in lines if line.startswith(('S-', 'T-', 'H-', 'A-'))]
