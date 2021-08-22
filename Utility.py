def peek_line(f):
    pos = f.tell()
    line = f.readline()
    f.seek(pos)
    return line