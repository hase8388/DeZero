import os
import subprocess


def _dot_var(v, verbose=False, color="orange"):
    """
    Variableをdot言語に変換
    """
    dot_var = '{} [label="{}", color = {}, style=filled]\n'
    name = "" if v.name is None else v.name
    if verbose and v.data is not None:
        if v.name is not None:
            name += ": "
        name += str(v.shape) + " " + str(v.dtype)

    return dot_var.format(id(v), name, color)


def _dot_func(f, color="lightblue"):
    """
    Functionをdot言語に変換
    """
    dot_func = '{} [label="{}", color = {}, style=filled, shape=box]\n'
    txt = dot_func.format(id(f), f.__class__.__name__, color)
    dot_edge = "{} -> {}\n"
    for x in f.inputs:
        txt += dot_edge.format(id(x), id(f))
    for y in f.outputs:
        txt += dot_edge.format(id(f), id(y()))
    return txt


def get_dot_graph(output, verbose=False):
    txt = ""
    funcs = []
    seen_set = set()

    def add_func(f):
        if f not in seen_set:
            funcs.append(f)
            seen_set.add(f)

    add_func(output.creator)
    txt += _dot_var(output, verbose)
    while funcs:
        func = funcs.pop()
        txt += _dot_func(func)
        for x in func.inputs:
            txt += _dot_var(x, verbose)
            if x.creator is not None:
                add_func(x.creator)

    return "digraph g {\n" + txt + "}"


def plot_dot_graph(output, verbose=True, to_file="graph.png"):
    dot_graph = get_dot_graph(output, verbose)
    # dotデータの保存用dir
    tmp_dir = os.path.join(os.path.expanduser("~"), "dezero")
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    graph_path = os.path.join(tmp_dir, "tmp_graph.dot")

    with open(graph_path, "w") as fh:
        fh.write(dot_graph)

    extension = os.path.splitext(to_file)[1][1:]  # 拡張子
    cmd = "dot {} -T {} -o {}".format(graph_path, extension, to_file)
    subprocess.run(cmd, shell=True)
    # jupyter で実行する場合、ディスプレイで表示
    try:
        from IPython import display

        return display.Image(filename=to_file)
    except:
        pass
