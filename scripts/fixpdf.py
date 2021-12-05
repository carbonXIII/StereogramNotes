#!/usr/bin/env python3

import sys
import os
import re
import tempfile
import subprocess

def replace(s, pattern, t):
    pattern = re.compile(pattern)
    off = 0
    while True:
        res = pattern.search(s, off)
        if not res:
            break

        off = res.start(1) + len(t)
        s = s[:res.start(1)] + t + s[res.end(1):]

    return s

def pdflatex(latex, name='images'):
    with open('{}.tex'.format(name), 'w') as f:
        f.write(latex)

    subprocess.run(['pdflatex', '{}.tex'.format(name)])
    return '{}.pdf'.format(name)

def convert_all(pdf, d='img'):
    subprocess.run(['rm', '-rf', d], capture_output=True)
    subprocess.run(['mkdir', '-p', d], capture_output=True)
    subprocess.run([
        'convert',
        '-density',
        '300',
        pdf,
        '{}/img-%0d.png'.format(d)
    ], capture_output=True)

    return ['{}/img-{}.png'.format(d, i) for i in range(len(os.listdir(d)))]

def pandoc_preprocess(tex, md):
    subprocess.run(['/home/shado/packages/latex-pandoc-preprocessor/preprocess.py', tex, md, 'no'])

def pandoc(md, html):
    subprocess.run(['pandoc',
                    '--bibliography', 'references.bib',
                    '--self-contained',
                    '-F', 'pandoc-crossref',
                    '--citeproc',
                    '--csl', 'https://raw.githubusercontent.com/citation-style-language/styles/master/ieee.csl',
                    '-V', 'fontsize=12pt',
                    '-V', 'mainfont:Arial',
                    md, '-o', html])

if __name__ == '__main__':
    with open(sys.argv[1], 'r') as f:
        contents = f.read()

    # copy 1
    only_images = contents

    # append preview headers
    #+LATEX_HEADER: \setlength\PreviewBorder{10pt}
    only_images = replace(only_images,
                          '\\\\documentclass.*?()\n',
                          '\n'.join([
                              '\\usepackage[active,graphics,tightpage]{preview}',
                              '\\PreviewEnvironment{figure}'
    ]))

    # set margin to 0
    # only_images = replace(only_images, '\\[margin=(.*?)\\]', '0in')

    # remove captions
    only_images = replace(only_images, '(\\\\caption.*?\n)', '')

    # generate a pdf with just our figures
    images_pdf = pdflatex(only_images)
    print(images_pdf)

    # generate pngs for each page
    images = convert_all(images_pdf)

    # copy 2
    final = contents

    # use only '$$' not '()'
    final = replace(final, r'(\\\()', '$')
    final = replace(final, r'(\\\))', '$')

    # replace figures with extracted images

    beg_p = re.compile('\\\\begin{figure}')
    end_p = re.compile('\\\\end{figure}')
    off = 0
    for image in images:
        beg = beg_p.search(final, off)
        end = end_p.search(final, beg.end())

        off = end.end()
        to_replace = final[beg.end():end.start()]

        caption = re.search('(\\\\caption.*?\n)', to_replace)[0]
        to_replace = '\n'.join([
            '[H]',
            '\\pdfimageresolution=300',
            '\\centerline{\\includegraphics{' + image + '}}',
            caption
        ])

        off = beg.end() + len(to_replace)
        final = final[:beg.end()] + to_replace + final[end.start():]

    with open('final.tex', 'w') as f:
        f.write(final)

    pandoc_preprocess('final.tex', 'final.md')
    pandoc('final.md', 'final.html')

