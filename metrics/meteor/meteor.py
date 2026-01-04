import os
import sys
import subprocess
import threading
import tarfile
import urllib.request

METEOR_GZ_URL = "http://aimagelab.ing.unimore.it/speaksee/data/meteor.tgz"
METEOR_JAR = "meteor-1.5.jar"


def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
    """
    Safer extractall để tránh path traversal.
    """
    for member in tar.getmembers():
        member_path = os.path.join(path, member.name)
        abs_directory = os.path.abspath(path)
        abs_target = os.path.abspath(member_path)
        if not abs_target.startswith(abs_directory):
            raise Exception("Unsafe tar file: path traversal detected")
    tar.extractall(path, members, numeric_owner=numeric_owner)


def download_from_url(url, save_path):
    """
    Download a file từ URL về local.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = downloaded / total_size * 100 if total_size > 0 else 0
        sys.stdout.write("\rDownloading: {:.2f}%".format(min(percent, 100)))
        sys.stdout.flush()

    print(f"Downloading from {url} to {save_path} ...")
    urllib.request.urlretrieve(url, save_path, reporthook=_progress)
    print("\nDownload completed!")


class Meteor:
    def __init__(self):
        base_path = os.path.dirname(os.path.abspath(__file__))
        jar_path = os.path.join(base_path, METEOR_JAR)
        gz_path = os.path.join(base_path, os.path.basename(METEOR_GZ_URL))

        # Ensure jar exists
        if not os.path.isfile(jar_path):
            if not os.path.isfile(gz_path):
                download_from_url(METEOR_GZ_URL, gz_path)
            with tarfile.open(gz_path, "r") as tar:
                safe_extract(tar, path=base_path)
            os.remove(gz_path)

        self.meteor_cmd = [
            "java", "-jar", "-Xmx2G", METEOR_JAR,
            "-", "-", "-stdio", "-l", "en", "-norm"
        ]
        self.meteor_p = subprocess.Popen(
            self.meteor_cmd,
            cwd=base_path,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,  # auto decode string
            bufsize=1
        )
        self.lock = threading.Lock()

    def compute_score(self, gts, res):
        """
        gts: dict[id] = list of references
        res: dict[id] = [hypothesis]
        """
        assert gts.keys() == res.keys()
        imgIds = gts.keys()
        scores = []

        eval_line = "EVAL"
        with self.lock:
            for i in imgIds:
                assert len(res[i]) == 1
                stat = self._stat(res[i][0], gts[i])
                eval_line += f" ||| {stat}"

            self._write_line(eval_line)

            for _ in imgIds:
                scores.append(float(self._read_line()))
            score = float(self._read_line())

        return score, scores

    def _stat(self, hypothesis_str, reference_list):
        # Clean input
        hypothesis_str = hypothesis_str.replace("|||", "").replace("  ", " ").strip()
        refs = " ||| ".join(reference_list)
        score_line = f"SCORE ||| {refs} ||| {hypothesis_str}"

        self._write_line(score_line)
        raw = self._read_line()
        numbers = [str(int(float(n))) for n in raw.split()]
        return " ".join(numbers)

    def _write_line(self, line: str):
        self.meteor_p.stdin.write(line + "\n")
        self.meteor_p.stdin.flush()

    def _read_line(self):
        return self.meteor_p.stdout.readline().strip()

    def __del__(self):
        try:
            with self.lock:
                if self.meteor_p:
                    self.meteor_p.stdin.close()
                    self.meteor_p.kill()
                    self.meteor_p.wait()
        except Exception:
            pass

    def __str__(self):
        return "METEOR"
