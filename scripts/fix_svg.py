#!/usr/bin/env python3

import sys
import os
import re

def process_svg(svg):
    bg = re.compile(r'<rect.*fill=\'(.*)\'/>')
    bg_match = bg.search(svg)
    svg = svg[:bg_match.start(1)] + 'none' + svg[bg_match.end(1):]

    fg = re.compile(r'<g.*fill=\'(.*)\'.*>')
    fg_match = fg.search(svg)
    svg = svg[:fg_match.start(1)] + '#000000' + svg[fg_match.end(1):]

    return svg

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

        with open('{}/svgs/{}.svg'.format(out_dir, base), 'w') as f:
            f.write(svg)

        mark_start = '@@comment: begin fixed {}@@'.format(base)
        mark_end = '@@comment: end fixed {}@@'.format(base)

        mark = re.compile(r'\n\n{}\n.*\n{}\n'.format(mark_start, mark_end))
        mark_match = mark.search(org)
        if mark_match is not None:
            org = org[:mark_match.start()] + org[mark_match.end():]

        embed = '\n\n{}\n[[file:svgs/{}.svg]]\n{}\n'.format(mark_start, base, mark_end)

        org = org[:match.end()] + embed + org[match.end():]

    with open(sys.argv[1], 'w') as f:
        f.write(org)
