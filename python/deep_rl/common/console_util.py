# The MIT License

# Copyright (c) 2017 OpenAI (http://openai.com)

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from collections import OrderedDict


def format_table(items):
    def _truncate(s):
        return s[:20] + '...' if len(s) > 23 else s

    key2str = OrderedDict()
    for (key, val) in items:
        if isinstance(val, float):
            valstr = '%-8.3g' % (val,)
        else:
            valstr = str(val)
        key2str[_truncate(key)] = _truncate(valstr)

    # Find max widths
    if len(key2str) == 0:
        print('WARNING: tried to write empty key-value dict')
        return
    else:
        keywidth = max(map(len, key2str.keys()))
        valwidth = max(map(len, key2str.values()))

    # Write out the data
    dashes = '-' * (keywidth + valwidth + 7)
    lines = [dashes]
    for (key, val) in key2str.items():
        lines.append('| %s%s | %s%s |' % (
            key,
            ' ' * (keywidth - len(key)),
            val,
            ' ' * (valwidth - len(val)),
        ))
    lines.append(dashes)
    return lines


def print_table(items):
    lines = format_table(items)
    for line in lines:
        print(line)


def print_progress(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 *
                                                     (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()
