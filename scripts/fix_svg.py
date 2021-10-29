#!/usr/bin/env python3

import sys
import os
import re
import subprocess

def process_svg(svg):
    bg = re.compile(r'<rect.*fill=\'(.*)\'/>')
    bg_match = bg.search(svg)
    svg = svg[:bg_match.start(1)] + 'none' + svg[bg_match.end(1):]

    fg = re.compile(r'<g.*fill=\'(.*?)\'.*>')
    off = 0
    while True:
        fg_match = fg.search(svg, off)
        if fg_match is None:
            break
        print(fg_match)

        off = fg_match.end()

        svg = svg[:fg_match.start(1)] + '#000000' + svg[fg_match.end(1):]

    path = re.compile(r'<path.*stroke=\'(.*?)\'.*>')
    off = 0
    while True:
        path_match = path.search(svg, off)
        if path_match is None:
            break
        print(path_match)

        off = path_match.end()

        svg = svg[:path_match.start(1)] + '#000000' + svg[path_match.end(1):]

    return svg

def convert_to_png(svg_path):
    subprocess.run(['inkscape', '--export-type=png', '--export-dpi=192', svg_path])

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Expected file as input')
        sys.exit(1)

    working_dir = os.path.dirname(sys.argv[1])
    with open(sys.argv[1], 'r') as f:
        org = f.read()

    out_dir = sys.argv[2]

    r = re.compile(r'#\+RESULTS:\s+\[\[file:badpngs/(.*).png\]\]')

    off = 0
    while True:
        match = r.search(org, off)
        if match is None:
            break

        off = match.end()

        print(match.group(1))
        base = match.group(1)

        with open('./{}/badpngs/{}.png'.format(working_dir, base), 'r') as f:
            svg = f.read()

        svg = process_svg(svg)

        svg_path = '{}/images/{}.svg'.format(out_dir, base)
        with open(svg_path, 'w') as f:
            f.write(svg)

        convert_to_png(svg_path)

        mark_start = '@@comment: begin fixed {}@@'.format(base)
        mark_end = '@@comment: end fixed {}@@'.format(base)

        mark = re.compile(r'\n\n{}\n.*\n{}\n'.format(mark_start, mark_end))
        mark_match = mark.search(org)
        if mark_match is not None:
            org = org[:mark_match.start()] + org[mark_match.end():]

        embed = '\n\n{}\n[[file:images/{}.png]]\n{}\n'.format(mark_start, base, mark_end)

        org = org[:match.end()] + embed + org[match.end():]

    with open(sys.argv[1], 'w') as f:
        f.write(org)
