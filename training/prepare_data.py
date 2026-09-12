"""Extract only the existing cleaned manifest and PNG images; never overwrite files."""
from pathlib import Path, PurePosixPath
import zipfile


def main():
    root = Path(__file__).resolve().parent.parent
    archive = root / 'data(2).zip'
    prefix = 'data/final/data/'
    with zipfile.ZipFile(archive) as z:
        entries = [i for i in z.infolist() if not i.is_dir() and (
            i.filename == prefix + 'dataset.csv' or
            (i.filename.startswith(prefix + 'images/') and i.filename.endswith('.png')))]
        if not any(i.filename == prefix + 'dataset.csv' for i in entries):
            raise ValueError('Archive is missing dataset.csv')
        targets = []
        for item in entries:
            relative = PurePosixPath(item.filename)
            target = root.joinpath(*relative.parts).resolve()
            if not target.is_relative_to(root) or '..' in relative.parts:
                raise ValueError(f'Invalid archive path: {item.filename}')
            if target.exists() and target.read_bytes() != z.read(item):
                raise FileExistsError(f'Existing file differs; no files overwritten: {target}')
            targets.append((item, target))
        for item, target in targets:
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('xb') as f:
                    f.write(z.read(item))
    print(f'Data ready: {root / prefix} ({len(entries) - 1} images)')


if __name__ == '__main__':
    main()
