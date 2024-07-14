import numpy as np
import matplotlib.pyplot as plt
from meteva.method.yes_or_no.score import pod_hfmc, sr_hfmc
import meteva
import os

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def performance(hfmc_array, grade_list, member_list=None,
                x_y="sr_pod", save_path=None, show=False,
                dpi=300, title="综合表现图",
                sup_fontsize=10,
                width=None, height=None,
                bias_list=[0.2, 0.4, 0.6, 0.8, 1, 1.25, 1.67, 2.5, 5],
                ts_list=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                ):
    '''
    Parameter
    ----------
    hfmc_array: 3D array ---> (n, len(grade_list), 4).  python numpy数组，
        n Fo.shape[0].表示n个模式的预测. n = len(member_list) if member_list is not None.
        len(grade_list). 表示阈值数量. len(grade_list)
        其中最后一维长度为4，分别记录了（命中数，空报数，漏报数，正确否定数）---> [TP,FP, FN, TN]
    grade_list: list
        阈值列表.the list of thresholds. eg: [0.1, 1, 5, 10, 20, 30, 40, 50, 60].
    member_list: list
        list of model name. 各预报成员的名称列表，缺省时系统由自动生成，将在bar图的legend中显示
    x_y: string
        xy轴坐标，缺省情况下会以成功率 命中率作为横纵坐标，当x_y != “sr_pod”时坐标会切换成far_mar.
    save_path: str
        图片保存路径，缺省时不输出图片，而是以默认绘图窗口形式展示.
    show: bool
        是否在屏幕显示图片，如果save_path 和save_dir 为都None时，程序内部会自动将show设置True.
    dpi: int
        绘图所采用dpi参数,效果同matplotlib中dpi参数.
    title: str
        图片标题，缺省时 为“综合表现图”.
    sup_fontsize:
        图片标题的字体大小，其它字体将根据标题字体大小自动设置，其中坐标轴字体大小 = sup_fontsize × 0.9, 坐标刻度的字体大小 = sup_fontsize × 0.8.
    width, height: float
        width:图片的宽度，缺省时程序自动设置. height:图片的高度，缺省时程序自动设置.
    bias_list: list
        the list of contour-bias line which will be plotted in photo.
    ts_list: list
        the list of contour-ts line which will be plotted in photo.
    Return
    ------
    sr, pod: array
        the sr and pod. --> shape: (n, len(grade_list)).
    '''
    pod = pod_hfmc(hfmc_array)
    sr = sr_hfmc(hfmc_array)
    leftw = 0.6
    rightw = 2
    uphight = 1.2
    lowhight = 1.2
    axis_size_x = 3.7
    axis_size_y = 3.5
    if width is None:
        width = axis_size_x + leftw + rightw

    if height is None:
        height = axis_size_y + uphight + lowhight

    fig = plt.figure(figsize=(width, height), dpi=dpi)
    ax1 = fig.add_axes([leftw / width, lowhight / width, axis_size_x / width, axis_size_y / height])

    x = np.arange(0.0001, 1, 0.0001)
    bias_list = [0.2, 0.4, 0.6, 0.8, 1, 1.25, 1.67, 2.5, 5] if bias_list is None else bias_list
    ts_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] if ts_list is None else ts_list
    for i in range(len(bias_list)):
        bias = bias_list[i]
        y1 = bias * x
        x2 = x[y1 < 1]
        y2 = y1[y1 < 1]
        if bias < 1:
            # bias <1 的 线
            ax1.plot(x2, y2, '--', color='k', linewidth=0.5)
            ax1.text(1.01, bias, "bias=" + str(bias), fontsize=sup_fontsize * 0.8)
        elif bias > 1:
            # bias》1的线
            ax1.plot(x2, y2, '--', color='k', linewidth=0.5)
            ax1.text(1.0 / bias - 0.05, 1.02, "bias=" + str(bias), fontsize=sup_fontsize * 0.8)
        else:
            # bias ==1 的线
            ax1.plot(x2, y2, '-', color='k', linewidth=0.5)

    for i in range(len(ts_list)):
        ts = ts_list[i]
        hf = 1
        x2 = np.arange(ts, 1, 0.001)
        hit = hf * x2
        hfm = hit / ts
        m = hfm - hf
        y2 = hit / (hit + m)
        # ts 的线
        plt.plot(x2, y2, "--", color="y", linewidth=0.5)
        error = np.abs(y2 - x2)
        index = np.argmin(error)
        sx = x2[index] + 0.02
        sy = y2[index] - 0.02
        ax1.text(sx, sy, "ts=" + str(ts), fontsize=sup_fontsize)

    new_sr = sr.reshape((-1, len(grade_list)))
    new_pod = pod.reshape((-1, len(grade_list)))

    new_sr_shape = new_sr.shape
    label = []
    legend_num = new_sr_shape[0]
    if member_list is None:
        if legend_num == 1:
            label.append('预报')
        else:
            for i in range(legend_num):
                label.append('预报' + str(i + 1))
    else:
        label.extend(member_list)

    colors = meteva.base.color_tools.get_color_list(legend_num)

    marker = ['o', 'v', 's', 'p', "P", "*", 'h', "X", "d", "1", "+", "x", ".", "^", "<", ">",
              "2", "3", "4", "8", "H", "D", "|", "_"]

    a_list = []
    grade_num = len(grade_list)
    if legend_num >= 1 and grade_num > 1:
        for line in range(legend_num):
            for i in range(len(grade_list)):
                ax1.plot(new_sr[line, i], new_pod[line, i], marker[i], label=i*line, color=colors[line], markersize=6)
                a_list.append(i*line)
        lines, label1 = ax1.get_legend_handles_labels()
        legend2 = ax1.legend(lines[0:len(lines):len(grade_list)], label, loc="upper right",
                             bbox_to_anchor=(1.5, 1), ncol=1, fontsize=sup_fontsize * 0.9)
        legend1 = ax1.legend(lines[:len(grade_list)], ['grade:'+str(i)for i in grade_list], loc="lower right",
                             bbox_to_anchor=(1.5, 0), ncol=1, fontsize=sup_fontsize * 0.9)
        ax1.add_artist(legend1)
        ax1.add_artist(legend2)
    elif legend_num > 1:
        for line in range(legend_num):
            i = 0
            ax1.plot(new_sr[line, i], new_pod[line, i], marker[line], label=i * line, color=colors[line], markersize=6)
            a_list.append(i * line)
        lines, label1 = ax1.get_legend_handles_labels()

        legend2 = ax1.legend(lines[0:len(lines):len(grade_list)], label, loc="upper right",
                             bbox_to_anchor=(1.5, 1), ncol=1, fontsize=sup_fontsize * 0.9)
        ax1.add_artist(legend2)

    elif grade_num > 1:
        colors = meteva.base.color_tools.get_color_list(grade_num)
        for i in range(grade_num):
            line = 0
            ax1.plot(new_sr[line, i], new_pod[line, i], marker[i], label=i * line, color=colors[i], markersize=6)
            a_list.append(i * line)
        lines, label1 = ax1.get_legend_handles_labels()

        legend1 = ax1.legend(lines[:len(grade_list)], ['grade:' + str(i) for i in grade_list], loc="upper right",
                             bbox_to_anchor=(1.5, 1), ncol=1, fontsize=sup_fontsize * 0.9)
        ax1.add_artist(legend1)

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    if x_y == "sr_pod":
        # ax1.set_xlabel("成功率", fontsize=sup_fontsize * 0.9)
        # ax1.set_ylabel("命中率", fontsize=sup_fontsize * 0.9)
        ax1.set_xlabel("SR", fontsize=sup_fontsize * 1.2)
        ax1.set_ylabel("POD", fontsize=sup_fontsize * 1.2)
    else:
        # ax1.set_xlabel("空报率", fontsize=sup_fontsize * 0.9)
        # ax1.set_ylabel("漏报率", fontsize=sup_fontsize * 0.9)
        ax1.set_xlabel("FAR", fontsize=sup_fontsize * 1.2)
        ax1.set_ylabel("MAR", fontsize=sup_fontsize * 1.2)
        x = np.arange(0, 1.01, 0.2)
        ax1.set_xticks(x)
        ax1.set_xticklabels(np.round(1 - x, 1))
        y = np.arange(0, 1.01, 0.2)
        ax1.set_yticks(y)
        ax1.set_yticklabels(np.round(1 - y, 1))

    title = title + "\n"
    ax1.set_title(title, fontsize=sup_fontsize)
    if save_path is None:
        show = True
    else:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print("检验结果已以图片形式保存至" + save_path)
    if show is True:
        plt.show()
    plt.close()

    return sr, pod
