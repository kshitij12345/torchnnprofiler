
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from itertools import cycle
import argparse
import copy


def horizontal_stacked_bar_plot(ax, species, weight_counts_matrix, text_matrix, color_dict):

    left = np.zeros(len(species))

    for i in range(weight_counts_matrix.shape[0]):
        weight_count = weight_counts_matrix[i]
        bar = ax.barh(species, weight_count, height=1, label=i,
                      left=left, edgecolor='black', linewidth=1)
        for j, rectangle in enumerate(bar):
            if text_matrix[i, j] != "":
                # Set color based on text label
                rectangle.set_color(color_dict[text_matrix[i, j]])
                width = rectangle.get_width()
                if text_matrix[i, j] != "":
                    text = text_matrix[i, j]

                    text_parts = text.split('.')
                    if text_parts[-1].isdigit():
                        text = '.'.join(text_parts[-2:])  # get 'aaa.0'
                    else:
                        # get 'bbb'
                        text = text_parts[-1].replace(
                            'remaining_cost', 'other')
                else:
                    text = ""
                ax.text(left[j] + width / 2, rectangle.get_y() + rectangle.get_height() / 2,
                        text, ha='center', va='center', rotation=90)
        left += weight_count
    return ax

def vertical_stacked_bar_plot(ax, species, weight_counts_matrix, text_matrix, color_dict):
    bottom = np.zeros(len(species))

    for i in range(weight_counts_matrix.shape[0]):
        weight_count = weight_counts_matrix[i]
        bar = ax.bar(species, weight_count, width=1, label=i,
                     bottom=bottom, edgecolor='black', linewidth=1)
        for j, rectangle in enumerate(bar):
            if text_matrix[i, j] != "":
                # Set color based on text label
                rectangle.set_color(color_dict[text_matrix[i, j]])
                height = rectangle.get_height()
                if text_matrix[i, j] != "":
                    text = text_matrix[i, j]

                    text_parts = text.split('.')
                    if text_parts[-1].isdigit():
                        text = '.'.join(text_parts[-2:])  # get 'aaa.0'
                    else:
                        # get 'bbb'
                        text = text_parts[-1].replace(
                            'remaining_cost', 'other')
                else:
                    text = ""
                ax.text(rectangle.get_x() + rectangle.get_width() / 2, bottom[j] + height / 2,
                        text, ha='center', va='center')
        bottom += weight_count


def visulize_the_profile_dict(profile_dict, time_flag='forward_cost', output_path='test.png', direction='vertical', figsize=(4, 8)):
    profile_dict = copy.deepcopy(profile_dict)
    for k, v in profile_dict.items():
        complete_ignore_cost(k, v)
    for k, v in profile_dict.items():
        add_remaining_cost(k, v)
    aaa = copy.deepcopy(profile_dict)
    first_key = list(profile_dict.keys())[0]
    for k, v in aaa.items():
        fill_in_to_deepth(k, v, get_max_depth(profile_dict[first_key]))
    species = {}
    weight_counts = {}

    for k, v in aaa.items():
        collect_species_weights(
            k, v, species, weight_counts, time_flag=time_flag)

    max_value = 0
    if isinstance(time_flag, str):
        time_flag = [time_flag]
    for flag in time_flag:
        max_value += profile_dict[first_key][flag]
    max_size = max([len(t) for t in weight_counts.values()])
    keys = np.sort(list(weight_counts.keys()))
    weight_counts_matrix = np.flip(np.stack([np.pad(np.array(
        weight_counts[i])/max_value*1000, (max_size-len(weight_counts[i]), 0)) for i in keys]).transpose(), 0)
    text_matrix = np.flip(np.stack([np.pad(np.array(
        species[i]), (max_size-len(species[i]), 0), constant_values="") for i in keys]).transpose(), 0)

    unique_labels = np.unique(text_matrix)
    colors = list(mcolors.TABLEAU_COLORS.values())
    color_cycle = cycle(colors)
    color_dict = {label: next(color_cycle) for label in unique_labels}

    # # Generate a palette with 500 unique colors
    # colors = sns.color_palette("husl", 500)

    # # Convert RGB values to hexadecimal color codes
    # hex_colors = [mcolors.rgb2hex(color) for color in colors]
    # color_dict = dict(zip(unique_labels, hex_colors))

    keys = np.arange(len(weight_counts_matrix))
    species = [f"Level{i}" for i in range(weight_counts_matrix.shape[1])]

    fig, ax = plt.subplots(figsize=figsize)
    if direction == 'vertical':
        vertical_stacked_bar_plot(
            ax, species, weight_counts_matrix, text_matrix, color_dict)
    elif direction == 'horizontal':
        horizontal_stacked_bar_plot(
            ax, species, weight_counts_matrix, text_matrix, color_dict)
    else:
        raise ValueError(
            f"direction must be 'vertical' or 'horizontal', but got {direction}")

    ax.set_title(
        f"{'+'.join(time_flag)}" if isinstance(time_flag, list) else f"{time_flag} ")
    ax.set_yticks([])
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.savefig(output_path, bbox_inches='tight')
    return fig, ax

def complete_ignore_cost(name,pool):
    if 'children' in pool and 'collect_flag' in pool and pool['collect_flag'] in ['list', 'dict']:
        #print(name)
        for key in ['forward_cost','backward_cost']:
            if key in pool:continue
            child_values = [child_value for child_key, child_value in pool['children'].items() if key in child_value]
            child_costs  = sum(v[key] for v in child_values )
            pool[key]    = child_costs
    if 'children' in pool:
        for key,val in pool['children'].items():
            complete_ignore_cost(key,val)

def add_remaining_cost(name,pool):
    if 'children' in pool:
        
        for key in ['forward_cost','backward_cost']:
            if key not in pool:continue
            total_cost = pool[key]
            child_values = [child_value for child_key, child_value in pool['children'].items() if key in child_value]
            child_costs = sum(v[key] for v in child_values )
            remaining_forward_cost = max(total_cost - child_costs,0)
            if remaining_forward_cost<1e-3:continue
            if f'{name}.remaining_cost' not in pool['children']:pool['children'][f'{name}.remaining_cost'] = {}
            pool['children'][f'{name}.remaining_cost'][key] = remaining_forward_cost
            pool['children'][f'{name}.remaining_cost']['deepth'] = child_values[0]['deepth']
        for key,val in pool['children'].items():
            add_remaining_cost(key,val)

def print_namespace_tree(namespace, indent=0):
    namespace = vars(namespace) if not isinstance(namespace, dict) else namespace
    for key, value in namespace.items():
        print(' ' * indent, end='')
        if isinstance(value, (dict, argparse.Namespace)):
            print(key)
            print_namespace_tree(value, indent + 4)
        else:
            print(f"{key:30s} ---> {value}")

def fill_in_to_deepth(key, pool, deepth):
    
    if 'children' in pool:
        for key, valuse in pool['children'].items():
            fill_in_to_deepth(key,valuse, deepth)
    else:
        ### repeat the structure into deepth
        if 'deepth' not in pool:
            #print(pool)
            return
        if pool['deepth'] < deepth:
            self = copy.deepcopy(pool)
            self['deepth']+=1
            pool['children'] = {key:self}
            fill_in_to_deepth(key,pool, deepth)


def collect_species_weights(key, pool, species, weight_counts, time_flag=['forward_cost']):
    if isinstance(time_flag, str):
        time_flag = [time_flag]
    if 'deepth' in pool:
        deepth = pool['deepth']
        keyaddQ = False
        for flag in time_flag:
            if flag in pool:
                if deepth not in species:
                    species[deepth] = []
                species[deepth].append(key)
                keyaddQ = True
            if keyaddQ:
                break
        valueaddQ = False
        value = 0
        for flag in time_flag:
            if flag in pool:
                if deepth not in weight_counts:
                    weight_counts[deepth] = []
                value += pool[flag]
                valueaddQ = True
        if valueaddQ:
            weight_counts[deepth].append(value)
    if 'children' not in pool:
        return
    for key, value in pool['children'].items():
        collect_species_weights(key, value, species,
                                weight_counts, time_flag=time_flag)


def get_max_depth(node):
    if 'children' in node and node['children']:
        return max(get_max_depth(child) for child in node['children'].values())
    else:
        return node['deepth']
