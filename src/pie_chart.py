import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import rgb_to_hsv

def luminance(color):
    r, g, b, _ = color
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

def clean_label(label):

    # print(label)

    temp = label.lower().replace("_"," ").replace("  ","")

    # print(temp)

    return temp
   
    # return label.replace("_", " ").lower().strip().replace("  ", "")

def count_files_in_subfolders(folder):
    
    subfolders = [f.path for f in os.scandir(folder) if f.is_dir()]
    counts = {}
    for subfolder in subfolders:
        files = [f for f in os.listdir(subfolder) if os.path.isfile(os.path.join(subfolder, f))]
        counts[os.path.basename(subfolder)] = len(files)
    return counts

def create_pie_chart(labels, sizes, title):

    cleaned_labels = [clean_label(label) for label in labels]

    cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, len(labels)))

    text_colors = ['white' if luminance(color) < 0.5 else 'black' for color in colors]
  
    # plt.figure(figsize=(6, 6))
    # plt.pie(sizes, labels=cleaned_labels, autopct=lambda pct: f'{pct:.1f}%', startangle=140)
    # # plt.axis('equal') 
    # plt.title(title)
    # plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
    # plt.savefig("./plots/"+title+'_pie.png')
    # plt.show()

    plt.figure(figsize=(4, 4))
    # plt.pie(sizes, labels=cleaned_labels, autopct='%1.1f%%', textprops={'fontsize': 8}, startangle=140)
    plt.pie(sizes, autopct='%1.0f%%', textprops={'fontsize': 8,'color': 'black'}, startangle=140,colors=colors,wedgeprops=dict(edgecolor='black', linewidth=0.10))
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title(title)
    # for i, (label, size) in enumerate(zip(cleaned_labels, sizes)):
    #     angle = (sum(sizes[:i]) + size / 2) / sum(sizes) * 360
    #     radius = 1.1  # Adjust the distance of the arrows from the center
    #     x = radius * np.cos(angle * np.pi / 180)
    #     y = radius * np.sin(angle * np.pi / 180)
    #     plt.annotate(label, (x, y), xytext=(1.2*x, 1.2*y), arrowprops=dict(arrowstyle="->"), fontsize='small')
    
    for text, color in zip(plt.gca().texts, text_colors):
        text.set_color(color)
    
    plt.legend(labels=cleaned_labels, loc='center', bbox_to_anchor=(0.85, 0.5), fontsize='small')
    # plt.tight_layout()
    # plt.savefig("./plots/"+title+"_pie.png", dpi=300)
    # plt.savefig("./plots/"+title+"_pie")
    # plt.show()
    # plt.gcf().canvas.manager.window.showMaximized()  # Maximize window (works on some systems)
    # fig = plt.gcf()
    # fig.set_size_inches(12, 9)
    # plt.savefig("./plots/"+title+"_pie.png")
    plt.show()
    # plt.close()
    


def plot_pie_charts_for_folders(folders):
   
    for folder in folders:
        counts = count_files_in_subfolders(folder)
        labels = list(counts.keys())
        sizes = list(counts.values())
        title = os.path.basename(folder)
        if folder == 'crop_disease':
            title = 'Crop Disease'
            
        elif folder == 'cassava':
            title = 'Cassava'
            
        elif folder == 'plant_village':
            title = 'Plant Village'
        create_pie_chart(labels, sizes, title)


folders = ['crop_disease', 'plant_village', 'cassava']


plot_pie_charts_for_folders(folders)
