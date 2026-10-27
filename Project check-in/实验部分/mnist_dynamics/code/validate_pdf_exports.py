"""Render the five final scientific PDFs locally for visual inspection."""
import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
FIGURES=('figure3d','figure3d_by_seed','figure3d_diagnostics','figure3e','reference_diagnostics')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    assert json.loads((ROOT/'results/qa.json').read_text())['complete']
    pdfinfo=shutil.which('pdfinfo.exe') or shutil.which('pdfinfo')
    pdftoppm=shutil.which('pdftoppm.exe') or shutil.which('pdftoppm')
    if not pdfinfo or not pdftoppm:raise RuntimeError('Local Poppler pdfinfo and pdftoppm are required.')
    workspace=next(p for p in ROOT.parents if (p/'log.md').exists())
    output=args.output or workspace/'tmp/pdfs/partc-final'
    output.mkdir(parents=True,exist_ok=True)
    checked=[]
    for stem in FIGURES:
        pdf=ROOT/'figures'/f'{stem}.pdf'
        info=subprocess.check_output([pdfinfo,pdf.name],cwd=pdf.parent,text=True,encoding='utf-8',errors='replace')
        pages=int(re.search(r'^Pages:\s+(\d+)',info,re.MULTILINE).group(1))
        assert pages==1,pdf.name
        # Relative ASCII arguments also work with the Unicode workspace path on Windows.
        prefix=os.path.relpath(output/stem,pdf.parent)
        subprocess.run([pdftoppm,'-png','-singlefile','-r','120',pdf.name,prefix],cwd=pdf.parent,check=True)
        png=output/f'{stem}.png'
        raw=png.read_bytes()
        assert raw[:8]==b'\x89PNG\r\n\x1a\n'
        width,height=struct.unpack('>II',raw[16:24])
        assert width>500 and height>300
        checked.append(dict(file=pdf.name,pages=pages,bytes=pdf.stat().st_size,
                            sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),
                            rendered_png=str(png),rendered_size=[width,height]))
    result=dict(render_passed=True,visual_inspection_required=True,pdfs=checked,
                pdfinfo=pdfinfo,pdftoppm=pdftoppm)
    (ROOT/'metadata/pdf_export_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(render_passed=True,pdfs=len(checked),output=str(output)),ensure_ascii=False))


if __name__=='__main__':main()
