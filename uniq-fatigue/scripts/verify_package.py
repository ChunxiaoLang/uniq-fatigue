import hashlib
from pathlib import Path

root=Path(__file__).resolve().parents[1]
count=0
for line in (root/"CHECKSUMS.sha256").read_text(encoding="utf-8").splitlines():
    digest,name=line.split("  ",1)
    path=(root/name).resolve()
    if not path.is_relative_to(root):raise ValueError("Checksum path outside package")
    actual=hashlib.sha256(path.read_bytes()).hexdigest()
    if actual!=digest:raise ValueError(f"Changed file: {name}")
    count+=1
print(f"Verified {count} files against CHECKSUMS.sha256")
