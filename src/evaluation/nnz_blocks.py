import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


columns = ['Dataset', 'Method', 'Number of nonzero block']

datasets = [('MAWI-226M', 136, 2116),
            ('MAWI-129M', 76, 676),
            ('MAWI-69M', 40, 196),
            ('MAWI-36M', 22, 64),
            ('MAWI-19M', 10, 16),
            ('GenBank-170M', 103, 1225),
            ('GenBank-55M', 37, 144),
            ('webbase-2001', 70, 576)]

raw_data = [
    ('19M', 'Arrow Decomposition', 10),
    ('19M', '1.5D Decomposition', 16),

    ('36M', 'Arrow Decomposition', 22),
    ('36M', '1.5D Decomposition', 64),

    ('69M', 'Arrow Decomposition', 40),
    ('69M', '1.5D Decomposition', 196),

    ('129M', 'Arrow Decomposition', 76),
    ('129M', '1.5D Decomposition', 676),

    ('226M', 'Arrow Decomposition', 136),
    ('226M', '1.5D Decomposition', 2116),

    ('170M', 'Arrow Decomposition', 37),
    ('170M', '1.5D Decomposition', 144),

    ('55M', '1.5D Decomposition', 1225),
    ('55M', 'Arrow Decomposition', 103,),

    ('117M', 'Arrow Decomposition', 70),
    ('117M', '1.5D Decomposition', 576),

]


# TODO Integrate this with arrow_block_count.py?

# Set variables
# sns.set()
sns.set(font_scale=1.2)
sns.set_style("whitegrid")
sns.color_palette("Greys", as_cmap=True)

data = pd.DataFrame.from_records(raw_data, columns=columns)

x = np.arange(len(datasets))  # the label locations
width = 0.5

fig, ax = plt.subplots(1, 1, figsize=(15, 6), sharey=True, sharex=True)

sns.barplot(x='Dataset', y='Number of nonzero block',
            hue='Method',
            data=data)
ax.set_xlabel(' ')
plt.yscale("log")
plt.savefig('nnz_blocks.pdf', bbox_inches='tight')