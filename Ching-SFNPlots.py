{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 122,
   "metadata": {
    "code_folding": [
     0
    ]
   },
   "outputs": [],
   "source": [
    "# Import statements\n",
    "import sys\n",
    "import os\n",
    "import pickle\n",
    "import pdb\n",
    "import re\n",
    "import h5py\n",
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "import seaborn as sns\n",
    "import pandas as pd\n",
    "from mpl_toolkits.mplot3d import Axes3D\n",
    "import matplotlib.patches as mpatches\n",
    "from itertools import combinations\n",
    "from plot_generation_script import *\n",
    "from scipy.stats import ttest_ind\n",
    "\n",
    "sns.set_style(\"white\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "datadir = \"/run/user/1000/gvfs/smb-share:server=typhos.local,share=data_01/NL/layerproject/processed/\"\n",
    "pattern = 'full_(IT|PT)(\\d+)_(\\d+)_.*\\.hdf5'"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Generate the E2 depths of each file"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [],
   "source": [
    "e2_dict = {}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {
    "code_folding": [
     0
    ]
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "E2 error in full_IT6_190206__data.hdf5\n",
      "E2 error in full_IT6_190218__data.hdf5\n",
      "E2 error in full_PT13_190123__data.hdf5\n",
      "E2 error in full_IT8_190206__data.hdf5\n",
      "E2 error in full_IT9_190307__data.hdf5\n",
      "E2 error in full_IT10_190319__data.hdf5\n",
      "E2 error in full_PT19_190718__data.hdf5\n",
      "E2 error in full_PT19_190720__data.hdf5\n",
      "E2 error in full_PT19_190801__data.hdf5\n"
     ]
    }
   ],
   "source": [
    "# Iterate through all files and populate the e2_dict\n",
    "def find_e2_neur(f):\n",
    "    \"\"\"\n",
    "    For some file f, will return the likely E2 combination. Returns indices\n",
    "    into the ens_neur array. So, the returned tuple will have values from 0,1,2,3\n",
    "    \"\"\"\n",
    "    \n",
    "    dff = np.array(f['dff'])\n",
    "    C = np.array(f['C'])\n",
    "    blen = f.attrs['blen']\n",
    "    trial_start = np.array(f['trial_start'])\n",
    "    trial_end = np.array(f['trial_end'])\n",
    "    cursor = np.array(f['cursor'])\n",
    "    ens_neur = np.array(f['ens_neur'])\n",
    "    exp_data = C\n",
    "    cursor = np.concatenate((np.zeros(blen), cursor)) # Pad cursor\n",
    "    if cursor.size > exp_data.shape[1]: # Experiment probably cut short\n",
    "        cursor = cursor[:exp_data.shape[1]]\n",
    "    elif cursor.size < exp_data.shape[1]: # Weird\n",
    "        print(\"Possible padding error in \" + f.filename)\n",
    "        padding = exp_data.shape[1] - cursor.size\n",
    "        cursor = np.concatentate(\n",
    "            (np.zeros(padding), cursor)\n",
    "            )\n",
    "        \n",
    "\n",
    "    # Generate all possible E2 combinations. We will find the combination\n",
    "    # with the maximal correlation value.\n",
    "    e2_possibilities = list(combinations(np.arange(ens_neur.size), 2))\n",
    "    best_e2_combo = None\n",
    "    best_e2_combo_val = 0.0\n",
    "    all_corrs = []\n",
    "\n",
    "    if len(e2_possibilities) == 1:\n",
    "        return np.array(e2_possibilities[0])\n",
    "\n",
    "    # Loop over each possible E2 combination and assign a score to it\n",
    "    for e2 in e2_possibilities:\n",
    "        mask = np.zeros(ens_neur.shape,dtype=bool)\n",
    "        mask[e2[0]] = True\n",
    "        mask[e2[1]] = True\n",
    "        e2_neur = ens_neur[mask]\n",
    "        e1_neur = ens_neur[~mask]\n",
    "        correlation = 0\n",
    "        for i in range(trial_end.size):\n",
    "            start_idx = trial_start[i]\n",
    "            end_idx = trial_end[i]\n",
    "            simulated_cursor = \\\n",
    "                np.sum(exp_data[e2_neur,start_idx:end_idx], axis=0) - \\\n",
    "                np.sum(exp_data[e1_neur,start_idx:end_idx], axis=0)\n",
    "            try:\n",
    "                trial_corr = np.nansum(\n",
    "                    cursor[start_idx:end_idx]*simulated_cursor\n",
    "                    )\n",
    "            except:\n",
    "                pdb.set_trace()\n",
    "            correlation += trial_corr\n",
    "        # If this is the best E2 combo so far, record it\n",
    "        all_corrs.append(correlation)\n",
    "        if correlation > best_e2_combo_val:\n",
    "            best_e2_combo_val = correlation\n",
    "            best_e2_combo = e2\n",
    "\n",
    "    # Write the most probable E2 combination to the H5 file\n",
    "    if ens_neur.size == 2:\n",
    "        best_e2_combo = np.arange(ens_neur.size)\n",
    "\n",
    "    # Sometimes we only get negative correlations, so we should return None\n",
    "    if best_e2_combo is None:\n",
    "        return None\n",
    "    return np.array([best_e2_combo[0], best_e2_combo[1]])\n",
    "\n",
    "for animaldir in os.listdir(datadir):\n",
    "    e2_dict[animaldir] = {}\n",
    "    animal_path = datadir + animaldir + '/'\n",
    "    if not os.path.isdir(animal_path):\n",
    "        continue\n",
    "    animal_path_files = os.listdir(animal_path)\n",
    "    animal_path_files.sort()\n",
    "    for file_name in animal_path_files:\n",
    "        result = re.search(pattern, file_name)\n",
    "        if not result:\n",
    "            continue\n",
    "        experiment_type = result.group(1)\n",
    "        experiment_animal = result.group(2)\n",
    "        experiment_date = result.group(3)\n",
    "        f = h5py.File(animal_path + file_name, 'r')\n",
    "        com_cm = np.array(f['com_cm'])\n",
    "        if 'e2_neur' not in f.keys():\n",
    "            e2_indices = find_e2_neur(f)\n",
    "            if e2_indices is None:\n",
    "                print(\"E2 error in \" + file_name)\n",
    "                continue\n",
    "        else:\n",
    "            e2_indices = np.array(f['e2_neur'])\n",
    "        e2_dict[animaldir][file_name] = e2_indices"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Plotting the distribution of labeled neurons (z-plane only)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "all_IT_zlocations = [] # Size: 8515\n",
    "all_PT_zlocations = [] # Size: 1596\n",
    "all_nonred_zlocations = [] # Size: 375029"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "code_folding": [
     0
    ]
   },
   "outputs": [],
   "source": [
    "for animaldir in os.listdir(datadir):\n",
    "    animal_path = datadir + animaldir + '/'\n",
    "    if not os.path.isdir(animal_path):\n",
    "        continue\n",
    "    for file_name in os.listdir(animal_path):\n",
    "        result = re.search(pattern, file_name)\n",
    "        if not result:\n",
    "            continue\n",
    "        experiment_type = result.group(1)\n",
    "        experiment_animal = result.group(2)\n",
    "        experiment_date = result.group(3)\n",
    "        f = h5py.File(animal_path + file_name, 'r')\n",
    "        redlabel = np.array(f['redlabel'])\n",
    "        com_cm = np.array(f['com_cm'])\n",
    "        red_zlocation = com_cm[redlabel, 2]\n",
    "        nonred_zlocation = com_cm[np.logical_not(redlabel), 2]\n",
    "        if experiment_type == \"IT\":\n",
    "            all_IT_zlocations.extend(red_zlocation)\n",
    "        elif experiment_type == \"PT\":\n",
    "            all_PT_zlocations.extend(red_zlocation)\n",
    "        all_nonred_zlocations.extend(nonred_zlocation)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "code_folding": [
     0
    ]
   },
   "outputs": [],
   "source": [
    "# Make dataframe pd_data\n",
    "all_IT_zlocations = np.array(all_IT_zlocations)\n",
    "all_PT_zlocations = np.array(all_PT_zlocations)\n",
    "all_nonred_zlocations = np.array(all_nonred_zlocations)\n",
    "all_zlocations = np.hstack((\n",
    "    all_IT_zlocations, all_PT_zlocations, all_nonred_zlocations\n",
    "    ))*-1\n",
    "all_labels = np.hstack((\n",
    "    ['IT']*len(all_IT_zlocations),\n",
    "    ['PT']*len(all_PT_zlocations),\n",
    "    ['Unlabeled']*len(all_nonred_zlocations)\n",
    "    ))\n",
    "pd_data = pd.DataFrame({\n",
    "    'Microns Below Cortical Surface': all_zlocations,\n",
    "    'Neuron Identity': all_labels\n",
    "    })"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 167,
   "metadata": {
    "code_folding": [
     1
    ]
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Figure size 720x720 with 0 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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q1alU6dOeHp68vDDDxMZGXlLS/H0798fHx8fypQpw/bt2wkMDKRDhw54enrSrl07qlevzmeffeY4vmPHjtx///2UKVOG1q1bO6Zt9PT05Pz58xw7dgyz2UydOnUoW7Zsrmq499576dSpE2azmbZt23L69GkGDx6Ml5cXTZs2xcvLyzFBeuPGjR1zZzz44INERUWxf/9+AD7++GNatGhBw4YN8fLyYujQoVkmBlq+fDnDhw/H398fLy8vhgwZwpYtW8jMzOTjjz+mefPmNGrUCC8vL4YNG5YvS0WJ+2kuCHGrmjVr0rx5c+bNm5dl/oKTJ0+SlJSUZbYyq9XKU089letz/3WKyBunWoTs0zbeOL3l9Q8I27dvT0pKCiNGjODixYs89dRTDB8+nFKlSrms4a/zNly/av7r9JClS5d2XAF//fXXzJgxg59++omMjAzS09MdE/Zcn3byr/X5+Pg4tk+dOsXgwYOzBKuHhwe///57tvfecccdWd4rhZcCWNxu6NChPP300/Tu3dvRds8999CoUSMWLVqU43tyM4XjX68QfX19OXXqVJb9p0+fdkwGdDOlSpViyJAhDBkyhBMnTjgW8+zSpYvL996KkSNH8o9//IMFCxZQunRpJk+e7Bgi8fX1zbIMUFpaWpYxXn9/f6ZMmZLjahy+vr6OpYUArl69muW9Unjp7xRxu8DAQNq2bct7773naGvevDnJycmsW7eOjIwMMjIySEpKcgSJqykcbxQWFkZycjIbNmwgMzOTTZs2ceTIEZo3b+6yvr179/LDDz9gtVopW7Ysnp6ejukp89Mff/xB+fLlKV26NElJSWzcuNGxLzIykk8//ZSDBw+Snp7Ov//9b/76kGq3bt144403OHnyJABnz551jB9HRkayfft2Dhw44HjvXyfDl8JLASwFYvDgwVnuCS5btiwLFy5k06ZNNGvWjKZNmzJjxgzHPLw9evSgVKlSPP7444wePZro6Oibnr9ChQrEx8ezaNEiGjduzIIFC4iPj6dixYoua0tNTWXo0KE0aNCAtm3bEhoaektDIbk1fvx4/v3vfxMSEsKcOXNo06aNY1+NGjV4+eWXGTFiBM2aNePOO++kYsWKjqkin3vuOcLDw+nduzchISE888wzJCUlOd4bGxvLCy+8QLNmzbjrrruyDElI4aW5IEQKoT/++INGjRqxZcsW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MYjGgAFrhhVzKeeD7/nxcIWrPkSk5FEAA+ZSZsZWHOvWPiafnezW84tI0eNyCOLo0aP06NGDdu3aAfD999/z1ltvub0wEZHizmUAv/zyy4wcOdKxjPyDDz7Ipk2b3F6YiEhx5zKAr169SnBwcJa268sEiYhI3rkM4AoVKvDrr786Vi7++OOPqVy5stsLExEp7lx+CDd+/HhefvllfvnlF5o1a8a9997L9OnTC6I2EZFizWUABwQEsHjxYq5cuYLNZqNs2bIFUZeISL7JtGbiaf5f3Ll67PjG493FZQ8zZ86kb9++3HXXXQBcuHCBd955h+HDh7u9OBGR/OBp9qTN621yffzm4ZvdWM3/uBwD3rFjhyN8AcqXL8+OHTvcWpSISEngMoCtVivp6emO7bS0tCzbIiKSNy6HIJ566il69OhBx44dMZlMrF69mg4dOhREbSIixZrLAO7Xrx81a9Zk79692O12Bg0aRLNmzQqiNhGRYi1XH/OFhYURFhbm7lpEREoUlwH81VdfMXHiRH755RcyMjKwWq14e3tz8ODBgqhPRKTYcvkhXFxcHDNnziQwMJCvv/6aSZMm0b1799vq9I033iA6Opr27dvTu3dvLBYLAHa7nUmTJhEREUF0dDTffvut4z1r166lVatWtGrVirVr195W/yIihUGuhiACAwOxWq2YzWY6depE165db6vTvn37EhMTA8CSJUuYM2cOcXFx7Nixg+TkZLZu3crXX3/NK6+8wsqVKzl//jyzZ89m9erVmEwmOnbsSHh4OOXLl7+tOgCOHDnCjz/+yPHSx2/7XK76CQoKcmsfIlK0uAxgb29v0tPTqV27NtOmTcPX15crV67cVqd/fZru6tWrjnkmEhMT6dChAyaTifr163Px4kXOnDnD/v37adKkCT4+PgA0adKEnTt3OqbIFBEpilwG8LRp07Db7cTGxrJ48WJOnz7NrFmzbrvj119/nXXr1lGuXDmWLFkCgMViwd/f33GMv78/FoslW7ufn59j2OJ2BQUFERQUxBd/fpEv57tZPyIif+U0gHv06MG7777LBx98wKhRoyhdujRDhgzJ9Yl79uxJampqtvaYmBhatmzJ8OHDGT58OHPnzmXp0qUMHToUu92e7XiTyeS0XUSkKHMawL/99hv79+/n008/JSoqKlsIPvzwwzc98eLFi3NVQLt27Rgw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4dj3+OOPU6FCBcd5/vOf/+Dp6ck333zj+O+SlpZGpUqVspxzz549OR6TlJREaGioY4y4bdu2JCcn38ZXn38UwCJkHQMW13x8fLhw4UKWtgsXLnDvvfcCOP7E9/DwwGq1Znv/m2++SePGjZkzZw4nTpzgueeec+y7MdxNJhN2u52nn36akSNHOq3J2TGffPKJ018YRtMQhIjcsjvvvJPKlSuzZ88eAM6fP8/OnTtp0KBBrt5/6dIl/Pz8AFi7dm2WfV988QXnz58nLS2NTz75hEceeYTHHnuMLVu28Pvvvzv6u/HK2tkxwcHB7N+/n3PnzpGRkcHHH398W197ftIVsIjkybRp05gwYQKvvfYaAIMHD+a+++7L1Xv79u3LSy+9xKJFi3j00Uez7GvQoAEvvvgix44dIzo6mrp16wIQExND7969sdlslCpVitjY2CyzHQYFBeV4TP369RkyZAhdu3alcuXKPPTQQ9hstnz6LtwezYYmImIQDUGIiBhEASwiYhAFsIiIQRTAIiIGUQCLiBhEASyGq1WrluNWJoCFCxcya9YsAyv6nzVr1hAXF5fjvpCQkNs6r8VicWyPHTvW8YhsfHx8ns8rRYsCWAzn5eXF1q1bOXv2bL6e1263F5r7PW+0du1azpw549iePHkyQUFBADedrlGKFz2IIYbz9PTk2Wef5d1332X48OFZ9p09e5bx48dz6tQpAMaMGUODBg2yTRHZrl07x5Vjv379aNy4MV999RVz5szh0KFDzJ07F7vdTlhYGKNGjQKuXcE+99xzfPbZZ5QpU4a33nqLu+++22mdx48f54UXXiAzM5NmzZpl2bdgwQI2b95Meno6ERERDB06lBMnTtCvXz8aNGjAoUOH8PPz46233mL79u188803vPDCC5QpU4bly5fTr18/XnzxRbZs2eKYlyIoKIiAgAAqVKhAjx49AHj99depVKlSlkd3pejSFbAUCn//+9/ZsGEDly5dytI+efJkevTowerVq5k1axbjxo1zea6jR4/SoUMH1q1bh6enJzNmzODdd99l3bp1HD58mE8++QSAK1euUK9ePT766CMaNmzIihUrbnreyZMn061bN1avXk3lypUd7bt27eLYsWOsWrWK9evX8+233/Lll18CcOzYMf7+97+TkJBAuXLl2LJlC61bt6ZOnTqOGb/KlCnjONf1UF6/fj3/+te/6Ny5M+vWrQPAZrORkJBAdHR07r6pUujpClgKhbJly9K+fXuWLFmSJZB2796dZfrAy5cvc/ny5Zueq0qVKtSvXx+4NrXhX2fCio6O5ssvv6Rly5aUKlWKFi1aAFCnTh2++OKLm5730KFDjrHp9u3bM2PGDODa3AVffPEFHTp0AK4Fe3JyMvfccw/33nsvtWvXBpzPDHYz9957Lz4+Pvz3v/8lNTWVhx56yDFTmBR9CmApNHr06EHHjh3p2LGjo81ms7F8+fIsoQxgNpuzjO/++eefjtd33HFHrvorVaqUY5YsZ7N23SinWbXsdjv9+/ena9euWdpPnDiRZeJvs9mcpc7c6tKlC2vWrCE1NZVOnTrd8vul8NIQhBQaPj4+tG7dmlWrVjnamjZtytKlSx3b3333HQBVq1blv//9LwDffvstJ06cyPGcwcHBfPnll5w9exar1UpCQgKNGjXKU30hISEkJCQA8NFHH2WpcfXq1fzxxx8AWCwWx4xcztx5552O42/k6emZZX7cli1bsnPnTg4fPkzTpk3zVLsUTgpgKVR69+7NuXPnHNtjx47lm2++ITo6mrZt27Js2TIAIiMjuXDhAu3bt2fZsmVUq1Ytx/P5+voyYsQIevToQfv27XnooYdo2bJlnmobO3YsH3zwAZ06dcoyDNK0aVPatWtH165diY6OZujQoU7D9bqnn36a8ePH0759e9LS0rLse+aZZ3jqqacc89p6eXnRuHFj2rRpk20JHinaNBuaSCFns9l4+umnefPNN53+opGiSVfAIoXYkSNHiIiI4LHHHlP4FkO6AhYRMYiugEVEDKIAFhExiAJYRMQgCmAREYMogEVEDKIAFhExyP8H+xWfd3KQz8IAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 360x360 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(10,10))\n",
    "sns.catplot(\n",
    "    x='Neuron Identity', y='Microns Below Cortical Surface',\n",
    "    data=pd_data, kind=\"boxen\", hue='Neuron Identity',\n",
    "    palette=dict(IT=\"darkmagenta\", PT=\"r\", Unlabeled=\"forestgreen\")\n",
    "    )\n",
    "plt.title(\"Depth Distribution of\\nNeurons Imaged\")\n",
    "plt.savefig('sfn_fig1.eps')\n",
    "plt.show(block=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Within-Session HPM Learning Plots, IT vs PT\n",
    "## Analysing HPM gain from baseline to BMI"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 165,
   "metadata": {
    "code_folding": [
     0
    ]
   },
   "outputs": [],
   "source": [
    "def plot_itpt_hpm():\n",
    "    \"\"\"\n",
    "    Aggregates hits per minute across all IT and PT animals. \n",
    "    Looks at max hpm in 10 minute windows.\n",
    "    \"\"\"\n",
    "\n",
    "    vals = []\n",
    "    ids = []\n",
    "    num_it = 0\n",
    "    num_pt = 0\n",
    "    \n",
    "    for animaldir in os.listdir(datadir):\n",
    "        animal_path = datadir + animaldir + '/'\n",
    "        if not os.path.isdir(animal_path):\n",
    "            continue\n",
    "        animal_path_files = os.listdir(animal_path)\n",
    "        animal_path_files.sort()\n",
    "        for file_name in animal_path_files:\n",
    "            result = re.search(pattern, file_name)\n",
    "            if not result:\n",
    "                continue\n",
    "            experiment_type = result.group(1)\n",
    "            experiment_animal = result.group(2)\n",
    "            experiment_date = result.group(3)\n",
    "            f = h5py.File(animal_path + file_name, 'r')\n",
    "            com_cm = np.array(f['com_cm'])\n",
    "            _, hpm, _, _ =\\\n",
    "                learning_params(\n",
    "                    experiment_type + experiment_animal,\n",
    "                    experiment_date,\n",
    "                    bin_size=1\n",
    "                    )\n",
    "            hpm_3min = np.convolve(hpm, np.ones((3,))/3, mode='valid')\n",
    "            max_hpm = np.max(hpm_3min)\n",
    "            hpm_gain = max_hpm - np.mean(hpm[:3])\n",
    "            if experiment_type == 'IT':\n",
    "                vals.append(hpm_gain)\n",
    "                ids.append(\"IT\")\n",
    "                num_it += 1\n",
    "            else:\n",
    "                vals.append(hpm_gain)\n",
    "                ids.append(\"PT\")\n",
    "                num_pt += 1 \n",
    "\n",
    "    # Plot some rectangles\n",
    "    df = pd.DataFrame({\n",
    "        'Values': vals, \"Neuron Identity\": ids\n",
    "        })\n",
    "    sns.barplot(\n",
    "        x='Neuron Identity', y='Values', data=df\n",
    "        )\n",
    "    plt.title('Gain in HPM from Experiment Beginning')\n",
    "    plt.savefig('sfn_fig2.eps')\n",
    "    plt.show(block=True)\n",
    "    with open(\"hpmgain.p\", \"wb\") as f:\n",
    "        pickle.dump(df, f)\n",
    "    \n",
    "    # T-Test\n",
    "    its = df.loc[df['Neuron Identity'] == 'IT']['Values']\n",
    "    pts = df.loc[df['Neuron Identity'] == 'PT']['Values']\n",
    "    tstat, pval = ttest_ind(its, pts)\n",
    "    print(\"T Test Results:\")\n",
    "    print(\"T-statistic = %d\"%tstat)\n",
    "    print(\"P-values = \" + str(pval))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 166,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "T Test Results:\n",
      "T-statistic = -3\n",
      "P-values = 0.0002647629502006049\n"
     ]
    }
   ],
   "source": [
    "plot_itpt_hpm()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Analysing the first ten minutes of all experiments"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 125,
   "metadata": {
    "code_folding": [
     0
    ]
   },
   "outputs": [],
   "source": [
    "def learning_params(\n",
    "    animal, day, sec_var='', bin_size=1, end_bin=None):\n",
    "    '''\n",
    "    Obtain the learning rate over time, including the fitted linear regression\n",
    "    model. This function also allows for longer bin sizes.\n",
    "    Inputs:\n",
    "        ANIMAL: String; ID of the animal\n",
    "        DAY: String; date of the experiment in YYMMDD format\n",
    "        BIN_SIZE: The number of minutes to bin over. Default is one minute\n",
    "    Outputs:\n",
    "        HPM: Numpy array; hits per minute\n",
    "        PERCENTAGE_CORRECT: float; the proportion of hits out of all trials\n",
    "        REG: The fitted linear regression model\n",
    "    '''\n",
    "    \n",
    "    folder_path = datadir + animal + '/'\n",
    "    f = h5py.File(\n",
    "        folder_path + 'full_' + animal + '_' + day + '_' +\n",
    "        sec_var + '_data.hdf5', 'r'\n",
    "        ) \n",
    "    if not os.path.exists(folder_path):\n",
    "        os.makedirs(folder_path)\n",
    "    fr = f.attrs['fr']\n",
    "    blen = f.attrs['blen']\n",
    "    blen_min = blen//600\n",
    "    hits = np.asarray(f['hits'])\n",
    "    miss = np.asarray(f['miss'])\n",
    "    array_t1 = np.asarray(f['array_t1'])\n",
    "    array_miss = np.asarray(f['array_miss'])\n",
    "    trial_end = np.asarray(f['trial_end'])\n",
    "    trial_start = np.asarray(f['trial_start'])\n",
    "    percentage_correct = hits.shape[0]/trial_end.shape[0]\n",
    "    bins = np.arange(0, trial_end[-1]/fr, bin_size*60)\n",
    "    [hpm, xx] = np.histogram(hits/fr, bins)\n",
    "    hpm = hpm[blen_min//bin_size:]\n",
    "    xx = -1*(xx[blen_min//bin_size]) + xx[blen_min//bin_size:]\n",
    "    xx = xx[1:]\n",
    "    if end_bin is not None:\n",
    "        end_frame = end_bin//bin_size + 1\n",
    "        hpm = hpm[:end_frame]\n",
    "        xx = xx[:end_frame]\n",
    "    tth = trial_end[array_t1] + 1 - trial_start[array_t1]\n",
    "\n",
    "    xx_axis = xx/(bin_size*60.0)\n",
    "    xx_axis = np.expand_dims(xx_axis, axis=1)\n",
    "    reg = LinearRegression().fit(xx_axis, hpm)\n",
    "    return xx_axis, hpm, percentage_correct, reg"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {
    "code_folding": [
     0
    ]
   },
   "outputs": [],
   "source": [
    "def plot_itpt_hpm(bin_size=1, plotting_bin_size=10, num_minutes=200):\n",
    "    \"\"\"\n",
    "    Aggregates hits per minute across all IT and PT animals. Performs regression\n",
    "    on the resulting data, and returns the p-value of how different linear\n",
    "    regression between the two animals are.\n",
    "    \"\"\"\n",
    "\n",
    "    # Getting all hits per minute arrays\n",
    "    IT_train = []\n",
    "    IT_target = []\n",
    "    PT_train = []\n",
    "    PT_target = []\n",
    "    num_it = 0\n",
    "    num_pt = 0\n",
    "    \n",
    "    for animaldir in os.listdir(datadir):\n",
    "        animal_path = datadir + animaldir + '/'\n",
    "        if not os.path.isdir(animal_path):\n",
    "            continue\n",
    "        hpm_arrays = []\n",
    "        for file_name in os.listdir(animal_path):\n",
    "            result = re.search(pattern, file_name)\n",
    "            if not result:\n",
    "                continue\n",
    "            experiment_type = result.group(1)\n",
    "            experiment_animal = result.group(2)\n",
    "            experiment_date = result.group(3)\n",
    "            f = h5py.File(animal_path + file_name, 'r')\n",
    "            xs, hpm, _, _ =\\\n",
    "                learning_params(\n",
    "                    experiment_type + experiment_animal,\n",
    "                    experiment_date,\n",
    "                    bin_size=bin_size\n",
    "                    )\n",
    "            xs = xs*bin_size\n",
    "            if experiment_type == 'IT':\n",
    "                for idx, x_val in enumerate(xs):\n",
    "                    if x_val <= num_minutes:\n",
    "                        IT_train.append(x_val)\n",
    "                        IT_target.append(hpm[idx])\n",
    "                num_it += 1\n",
    "            else:\n",
    "                for idx, x_val in enumerate(xs):\n",
    "                    if x_val <= num_minutes:\n",
    "                        PT_train.append(x_val)\n",
    "                        PT_target.append(hpm[idx])\n",
    "                num_pt += 1\n",
    "\n",
    "    # Collect data\n",
    "    IT_train = np.array(IT_train).squeeze()\n",
    "    IT_target = np.array(IT_target)\n",
    "    PT_train = np.array(PT_train).squeeze()\n",
    "    PT_target = np.array(PT_target)\n",
    "    fig = plt.figure()\n",
    "    ax = fig.add_subplot(111)\n",
    "\n",
    "    # p-val for linear regression slope similarity\n",
    "    p_val = linreg_pval(IT_train, IT_target, PT_train, PT_target)\n",
    "    print(\"Comparing linear regression slopes of IT and PT:\")\n",
    "    print(\"p-val = \" + str(p_val))\n",
    "\n",
    "    # Some options:\n",
    "    # Order 1, Order 2, Logx True\n",
    "    sns.regplot(\n",
    "        IT_train, IT_target,\n",
    "        x_bins=plotting_bin_size,\n",
    "        color='lightseagreen', label='IT (%d Experiments)'%num_it\n",
    "        )\n",
    "    sns.regplot(\n",
    "        PT_train, PT_target,\n",
    "        x_bins=plotting_bin_size,\n",
    "        color='coral', label='PT (%d Experiments)'%num_pt\n",
    "        )\n",
    "    ax.set_ylabel('Number of Hits')\n",
    "    ax.set_xlabel('Minutes into the Experiment')\n",
    "    plt.title('Hits/%d-min of All Experiments'%bin_size)\n",
    "    plt.legend()\n",
    "    plt.show(block=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "code_folding": [
     0
    ],
    "collapsed": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Comparing linear regression slopes of IT and PT:\n",
      "p-val = [0.53464743]\n"
     ]
    },
    {
     "data": {
      "image/png": "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P88999wTP86yLO644w6efvppvF4vZ599Nueddx5///vfufPOO/F6vfh8Pq6++momTZrUZVx77LEHu+3W2Yf6hBNO4Dvf+U6frjUTzz//PB9//HFOP+PVV18lEAhwwAEHpD1Ga82cOXO49dZbczYOEFEQhJJnyaaNXPDaS1iWRVs0So3Px03LlzLnoMM5cPiIPr23UoonnngCgOnTp3PkkUcyderUlMf+9a9/ZcqUKfjsgpIXXngh7e3t/PnPf044bt68eaxdu5a//e1veL1eNm0y1fIPPfRQjjnmGDweDx988AGXXnopTz/dtTtvVVVVfEy5JhKJcMwxx3DMMcfk9HNee+01ampqMoqCUop169axZs0axo4dm7OxiChkSUskzNZQCICtoRAtkXCfV2KC0FdaImEueO0lWiOR+HNt0SgAF7z2Eq9OOYlaf37+zZ988kl+8YtfxB8feuihvPrqq12O+9Of/sTtt9+O1+5JMnz4cABqa2vjx7S3t/fINr59+3ZOP/10Zs+ezU477cRll13GIYccwplnnsn+++/P1772NV599VUGDRrEr371K4YNG8aqVau4/vrr2bJlC1VVVdx4443svPPOTJ8+ncGDB7N8+XL22msvdtttN9577z2uvfZapk+fTmVlJStWrGDNmjXMmjWL+fPn8/bbb7PvvvvGV/Evv/wyv/71rwmFQjQ2NjJr1ixqa2s5+uijOfXUU3nhhReIRCLccccdVFZWMnfuXLxeLwsWLGDmzJls2LCBu+++G6/XS319PQ8//DAARx11FE899RTf/va3e/4HyhLxKWTBkk0bOeS5hazraAdgXUc7hzy3kCWbNhZ4ZMJAZ+HqJiwrufW5wbIsFq5pyss4QqEQTU1NNDQ0dHtsU1MTixYt4rTTTuPCCy/kk08+ib/23HPPMXXqVL773e9yyy23pDy/o6ODU045JX5btGgR9fX1XHvttcyYMYOnnnqKzz77jDPPPBOAtrY29txzT+bPn8+BBx7Ib37zGwBmzpzJzJkzmTdvHldddRXXX399/DM++eQTHnjgAaZPn97l87dt28ZDDz3EjBkzuOiiizj//PN56qmn+PDDD3n//ffZvHkzs2fPZs6cOcyfP5+9996bOXPmxM8fOnQo8+fP56yzzuL++++noaGBs846i/PPP58nnniCSZMm8dvf/pb77ruPBQsWMHv27Pi5e++9N2+88Ua3v+O+IDuFbki1ErOA1kgk7ysxQUjmk9aW+M4gmbZolJUt2/Myji1btlBfX5/VsaFQiMrKSubNm8ezzz7L1VdfzSOPPALAlClTmDJlCkuWLOHOO+/s4reA9OajyZMn8/TTT3PDDTckvO71ejn++OMBOOWUU7j44otpbW3lrbfe4pJLLkkYl8PUqVPjZrBkjjrqKDweD0opRowYgVIKgF122YXVq1ezbt06PvroI84++2zAJCLut99+8fOPO+44wEzwzz33XMrP2H///Zk+fTpf/vKXmTJlSvz54cOHs379+pTn9Bcym3VDNiuxr43vXZKIIPSVibV11Ph8KYWhxudjQl12E3VfqaqqSphUMzF69Oj4xDhlyhRmzJjR5ZgDDzyQVatWsXnzZoYNG5bV+8ZiMT7++GMqKyvZunUrO+ywQ8rjPB4PlmUxaNCgtL6J6urqtJ9TUVERfx/nPhjxiUQieL1eJk+ezC9/+cuU5zuZxl6vl2gaQb/hhhtYunQpixcv5tRTT+Xxxx9n6NChBINBKisr046tPxDzUTcUy0pMEFJx4rjGtLZ3j8fDiWMb8zKOwYMHE41GCQaD3R577LHH8p///AcwDtaJEycCsHLlyvgCbNmyZYTDYYYOHZr1GB544AF23nlnfvnLX3L11VcTDptmWLFYjGeeeQYwfo/Pf/7z1NXV0dDQwN/+9jfALPA++OCDrD8rE/vttx9vvvkmK1euBIx/5H//+1/Gc2pra2ltbY0/XrVqFfvuuy+XXHIJQ4cOZd26dYAxa+266679Ms50yE6hG4plJSYIqajzB5hz0OFdoo88Hg9zDjo8r6bNyZMn88Ybb/CFL3wBgHPOOYcVK1bQ1tbGEUccwc0338zhhx/Od77zHa644goefPBBampquPnmmwF45plneOKJJ/D7/VRVVfGrX/0qpeA5PgWHww8/nK9+9as8+uijPProo9TV1XHggQcye/ZsfvSjH1FTU8N///tfTjvtNOrq6rjjjjsAuO222/jZz37G7NmziUQiHH/88ey+++59/j0MGzaMWbNmcdlll8V3T5deemnGshNHHXUUP/rRj3j++eeZOXMmDzzwQFwkDznkkPi4Xn31VY488sg+jzETnnSmkWLltNNOs+bNm5e3z2uJhDnkuYUJPgWHWr9ffApCTnj//ffZY489sj6+NRJh4ZomVrZsZ0JdPSeObcz793L58uXMmTOH2267La+f2x37778/b731VqGH0WdCoRDf+MY3eOSRR/B387dN9f1RSr2hte6a+JGEzGbd4F6JtUUiWIAHqPH7874SE4R01Pr9Bfdt7bnnnhx88MFEo9G0Tlqh96xZs4bLL7+8W0HoK+JTyIIDh4/g1SknsUOVcT7tUFXNq1NO6nNikCCUG6effnrRCUI57BIAJk6cyMEHH5zzzxFRyJJav58hdqTBkIoK2SEIglCWiCgIgiAIcUQUBEEQhDgiCoIgCEIcMYwLQjkQbIf3XobNa2HYGNj7MKhMn5WbDU6J6mg0yk477cRPf/rTePnojRs34vV649nGjz76aEJ2r2VZfPOb3+S3v/0tdXV1zJgxg8WLFzN8+HAWLlwYP+7nP/85L7zwAoFAgPHjxzNr1iwGDRoEwAcffMB11+GOMiUAABDzSURBVF1HS0sLXq+Xxx57rEs277nnnsv69eupqqoCYMKECdx11119uu5MfPrpp9x88805/Yzm5mbeeustTjrppLTHhEIhLrjgAh588MF+j0bKqSgopaYCdwI+4F6t9a1Jr58P3Aastp/6jdb63lyOSRDKjpXL4eGbwIpBOAiBSnhmDnz9GpjQ+0ZC7hpDl19+OYsWLYo//vWvf01NTQ3Tpk1Lee4///lPdt99d+rq6gA47bTT+MY3vsFVV12VcNzkyZPjYZa33XYb99xzD1deeSWRSIQrr7yS2267jd13350tW7aknfx+8Ytf8LnPfa7X15ktkUiE0aNH51QQAFavXs3ChQszikJFRQWHHnooixYt4uSTT+7Xz8+ZKCilfMDdwBSgGViilFqgtV6edOiftdYX52ocglDWBNuNIITaO58L26UmHr4JLr+vzzsGgEmTJqG1zvr4J598Ml6lFEwto+bm5i7HHXbYYfH7++23X7x/wiuvvIJSKp7J25NyFwDf+973+NKXvsSpp57K3LlzWbJkCbfffjvnnnsuu+++O++++y4tLS3ccsst7LPPPrS1tXHjjTfy4YcfEo1Gufjiizn22GOZN28eixcvJhQK0dbWxi233MJFF13EwoULmTdvHn//+9+JxWJ8+OGHfOtb3yIcDvPEE09QUVHB73//e4YMGZKxRHddXR3vvfceGzZs4Morr2Tq1KncfvvtfPzxx5xyyil8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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plot_itpt_hpm(\n",
    "    bin_size=1, plotting_bin_size=10, num_minutes=100\n",
    "    )"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Analysing the first ten minutes of the first ten sessions of each mouse"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {
    "code_folding": [
     0
    ]
   },
   "outputs": [],
   "source": [
    "def plot_itpt_hpm(bin_size=1, plotting_bin_size=10, num_minutes=200, first_N_experiments=5):\n",
    "    \"\"\"\n",
    "    Aggregates hits per minute across all IT and PT animals. Performs regression\n",
    "    on the resulting data, and returns the p-value of how different linear\n",
    "    regression between the two animals are.\n",
    "    \"\"\"\n",
    "\n",
    "    # Getting all hits per minute arrays\n",
    "    IT_train = []\n",
    "    IT_target = []\n",
    "    PT_train = []\n",
    "    PT_target = []\n",
    "    num_it = 0\n",
    "    num_pt = 0\n",
    "    \n",
    "    for animaldir in os.listdir(datadir):\n",
    "        animal_path = datadir + animaldir + '/'\n",
    "        if not os.path.isdir(animal_path):\n",
    "            continue\n",
    "        animal_path_files = os.listdir(animal_path)\n",
    "        animal_path_files.sort()\n",
    "        animal_path_files = animal_path_files[:first_N_experiments]\n",
    "        for file_name in animal_path_files:\n",
    "            result = re.search(pattern, file_name)\n",
    "            if not result:\n",
    "                continue\n",
    "            experiment_type = result.group(1)\n",
    "            experiment_animal = result.group(2)\n",
    "            experiment_date = result.group(3)\n",
    "            f = h5py.File(animal_path + file_name, 'r')\n",
    "            xs, hpm, _, _ =\\\n",
    "                learning_params(\n",
    "                    experiment_type + experiment_animal,\n",
    "                    experiment_date,\n",
    "                    bin_size=bin_size\n",
    "                    )\n",
    "            xs = xs*bin_size\n",
    "            if experiment_type == 'IT':\n",
    "                for idx, x_val in enumerate(xs):\n",
    "                    if x_val <= num_minutes:\n",
    "                        IT_train.append(x_val)\n",
    "                        IT_target.append(hpm[idx])\n",
    "                num_it += 1\n",
    "            else:\n",
    "                for idx, x_val in enumerate(xs):\n",
    "                    if x_val <= num_minutes:\n",
    "                        PT_train.append(x_val)\n",
    "                        PT_target.append(hpm[idx])\n",
    "                num_pt += 1\n",
    "\n",
    "    # Collect data\n",
    "    IT_train = np.array(IT_train).squeeze()\n",
    "    IT_target = np.array(IT_target)\n",
    "    PT_train = np.array(PT_train).squeeze()\n",
    "    PT_target = np.array(PT_target)\n",
    "    fig = plt.figure()\n",
    "    ax = fig.add_subplot(111)\n",
    "\n",
    "    # p-val for linear regression slope similarity\n",
    "    p_val = linreg_pval(IT_train, IT_target, PT_train, PT_target)\n",
    "    print(\"Comparing linear regression slopes of IT and PT:\")\n",
    "    print(\"p-val = \" + str(p_val))\n",
    "\n",
    "    # Some options:\n",
    "    # Order 1, Order 2, Logx True\n",
    "    sns.regplot(\n",
    "        IT_train, IT_target,\n",
    "        x_bins=plotting_bin_size,\n",
    "        color='lightseagreen', label='IT (%d Experiments)'%num_it\n",
    "        )\n",
    "    sns.regplot(\n",
    "        PT_train, PT_target,\n",
    "        x_bins=plotting_bin_size,\n",
    "        color='coral', label='PT (%d Experiments)'%num_pt\n",
    "        )\n",
    "    ax.set_ylabel('Number of Hits')\n",
    "    ax.set_xlabel('Minutes into the Experiment')\n",
    "    plt.title('Hits/%d-min of All Experiments'%bin_size)\n",
    "    plt.legend()\n",
    "    plt.show(block=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {
    "code_folding": [
     0
    ],
    "collapsed": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Comparing linear regression slopes of IT and PT:\n",
      "p-val = [0.00637738]\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plot_itpt_hpm(\n",
    "    bin_size=1, plotting_bin_size=10, num_minutes=10,\n",
    "    first_N_experiments=10\n",
    "    )"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Analysing the first ten minutes of the last ten sessions of each mouse"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "code_folding": [
     0
    ]
   },
   "outputs": [],
   "source": [
    "def plot_itpt_hpm(bin_size=1, plotting_bin_size=10, num_minutes=200, last_N_experiments=5):\n",
    "    \"\"\"\n",
    "    Aggregates hits per minute across all IT and PT animals. Performs regression\n",
    "    on the resulting data, and returns the p-value of how different linear\n",
    "    regression between the two animals are.\n",
    "    \"\"\"\n",
    "\n",
    "    # Getting all hits per minute arrays\n",
    "    IT_train = []\n",
    "    IT_target = []\n",
    "    PT_train = []\n",
    "    PT_target = []\n",
    "    num_it = 0\n",
    "    num_pt = 0\n",
    "    \n",
    "    for animaldir in os.listdir(datadir):\n",
    "        animal_path = datadir + animaldir + '/'\n",
    "        if not os.path.isdir(animal_path):\n",
    "            continue\n",
    "        animal_path_files = os.listdir(animal_path)\n",
    "        animal_path_files.sort()\n",
    "        animal_path_files = animal_path_files[-last_N_experiments:]\n",
    "        for file_name in animal_path_files:\n",
    "            result = re.search(pattern, file_name)\n",
    "            if not result:\n",
    "                continue\n",
    "            experiment_type = result.group(1)\n",
    "            experiment_animal = result.group(2)\n",
    "            experiment_date = result.group(3)\n",
    "            f = h5py.File(animal_path + file_name, 'r')\n",
    "            xs, hpm, _, _ =\\\n",
    "                learning_params(\n",
    "                    experiment_type + experiment_animal,\n",
    "                    experiment_date,\n",
    "                    bin_size=bin_size\n",
    "                    )\n",
    "            xs = xs*bin_size\n",
    "            if experiment_type == 'IT':\n",
    "                for idx, x_val in enumerate(xs):\n",
    "                    if x_val <= num_minutes:\n",
    "                        IT_train.append(x_val)\n",
    "                        IT_target.append(hpm[idx])\n",
    "                num_it += 1\n",
    "            else:\n",
    "                for idx, x_val in enumerate(xs):\n",
    "                    if x_val <= num_minutes:\n",
    "                        PT_train.append(x_val)\n",
    "                        PT_target.append(hpm[idx])\n",
    "                num_pt += 1\n",
    "\n",
    "    # Collect data\n",
    "    IT_train = np.array(IT_train).squeeze()\n",
    "    IT_target = np.array(IT_target)\n",
    "    PT_train = np.array(PT_train).squeeze()\n",
    "    PT_target = np.array(PT_target)\n",
    "    fig = plt.figure()\n",
    "    ax = fig.add_subplot(111)\n",
    "\n",
    "    # p-val for linear regression slope similarity\n",
    "    p_val = linreg_pval(IT_train, IT_target, PT_train, PT_target)\n",
    "    print(\"Comparing linear regression slopes of IT and PT:\")\n",
    "    print(\"p-val = \" + str(p_val))\n",
    "\n",
    "    # Some options:\n",
    "    # Order 1, Order 2, Logx True\n",
    "    sns.regplot(\n",
    "        IT_train, IT_target,\n",
    "        x_bins=plotting_bin_size,\n",
    "        color='lightseagreen', label='IT (%d Experiments)'%num_it\n",
    "        )\n",
    "    sns.regplot(\n",
    "        PT_train, PT_target,\n",
    "        x_bins=plotting_bin_size,\n",
    "        color='coral', label='PT (%d Experiments)'%num_pt\n",
    "        )\n",
    "    ax.set_ylabel('Number of Hits')\n",
    "    ax.set_xlabel('Minutes into the Experiment')\n",
    "    plt.title('Hits/%d-min of All Experiments'%bin_size)\n",
    "    plt.legend()\n",
    "    plt.show(block=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {
    "code_folding": [
     0
    ],
    "collapsed": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Comparing linear regression slopes of IT and PT:\n",
      "p-val = [0.43208433]\n"
     ]
    },
    {
     "data": {
      "image/png": "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l17i1ltvxbZtGhoauPLKK2lsbOSZZ57h05/+NHfccQdXXnklP/rRj5L7rFixguuvv57ly5dz3333JWcyF1xwAQ899BA33HADr7/+OrfccgtvvfVW8rnvfe97XHfdddx6661s376dW265hSuuuIKXX34ZUELyne98h49+9KNce+21/O53vwNg06ZN3HLLLXzuc5/jAx/4AHfddRe2bfPYY4/R0NDA3/3d33HLLbdgWRZf/vKXueaaa7j22mt55JFHAGV70traSmNj48B+IRrNmYYZg5OtcKIJotk3PCyWktQ7gD/l+yRtZozbNq+l3ezsLd7hXPRu27yWTcuupcyT/48kGo1y+PBhxo9XOnjBBRcwf/58Fi5ciG3b3HzzzUydOpXjx4+zf/9+amtrqamp4eWXXyYWi2U85sqVK9m6dWvy8e9//3uWLVvGiy++yOOPP87atWv57Gc/S3W18sN66623eO655ygpKeH6669n8eLFlJaW8qc//Ynf/va3eL1evvGNb/Dcc8/x4Q9/mI6ODqZNm8add955yrk7Ojq4+OKL+dKXvsRnPvMZfvCDH/CrX/2Kffv2cffdd3PFFVfw1FNPUVFRwdNPP000GuWmm25iwYIFAOzatYsXXniBYDDIxz/+cbZu3conP/lJHnnkER599FFGjRrFjh07qK+v5/nnnwfgxIkTyfPPmjWLbdu2ceWVV+bmF6TRnM5YJoTaIdIO/WiNUHBREEIsRYnCwnyf6/kjh+muf4Rt2zz/3mE+NjH/tewtLS1UVFQkHx86dIh9+/axZo3qcX777bezZcsW5s2bxze+8Q2+8IUv4HK5uOCCCzh8+HDGY3YXPrr33nu55pprOP/887nmmmuS2y+77DJGjhwJwLJly9i6dSsej4cdO3Zw/fXXA52lvgBut7vbi67X6+Xyyy8HYPr06fh8PrxeL9OnT+fIkSMArF+/HillcqZz8uRJDh06hNfr5dxzz6WmpgaAGTNmcOTIEebOnZt2jgkTJnD48GG++c1vsnjxYhYu7PxzqayspKGhIePYNJqiYN+bsPFZuGw5nHVufs5hWRB2xCAe7/dhCioKQohzgV8AH5RSNuf7fAfb25Izg650WBaH2k7mewiAWmEbjUaTj1966SXOO+88ysrKAFi0aBFvvPEG8+bN433vex/ve9/7AHX373L1LeJXX1+Py+Wi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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plot_itpt_hpm(\n",
    "    bin_size=1, plotting_bin_size=10, num_minutes=10,\n",
    "    last_N_experiments=10\n",
    "    )"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Within-Session HPM Learning Plots, IT shallow vs IT deep vs PT\n",
    "## Analysing HPM gain from baseline to BMI"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 120,
   "metadata": {
    "code_folding": [
     0
    ]
   },
   "outputs": [],
   "source": [
    "def plot_itpt_hpm():\n",
    "    \"\"\"\n",
    "    Aggregates hits per minute across all IT and PT animals. \n",
    "    Looks at max hpm in 10 minute windows.\n",
    "    \"\"\"\n",
    "\n",
    "    vals = []\n",
    "    ids = []\n",
    "    num_itshallow = 0\n",
    "    num_itdeep = 0\n",
    "    num_pt = 0\n",
    "    \n",
    "    for animaldir in os.listdir(datadir):\n",
    "        animal_path = datadir + animaldir + '/'\n",
    "        if not os.path.isdir(animal_path):\n",
    "            continue\n",
    "        animal_path_files = os.listdir(animal_path)\n",
    "        animal_path_files.sort()\n",
    "        for file_name in animal_path_files:\n",
    "            result = re.search(pattern, file_name)\n",
    "            if not result:\n",
    "                continue\n",
    "            experiment_type = result.group(1)\n",
    "            experiment_animal = result.group(2)\n",
    "            experiment_date = result.group(3)\n",
    "            f = h5py.File(animal_path + file_name, 'r')\n",
    "            com_cm = np.array(f['com_cm'])\n",
    "            try:\n",
    "                e2_indices = e2_dict[animaldir][file_name]\n",
    "            except:\n",
    "                continue\n",
    "            ens_neur = np.array(f['ens_neur'])\n",
    "            e2_neur = ens_neur[e2_indices]\n",
    "            e2_depths = np.mean(com_cm[e2_neur,2])\n",
    "            _, hpm, _, _ =\\\n",
    "                learning_params(\n",
    "                    experiment_type + experiment_animal,\n",
    "                    experiment_date,\n",
    "                    bin_size=1\n",
    "                    )\n",
    "            hpm_5min = np.convolve(hpm, np.ones((5,))/5, mode='valid')\n",
    "            max_hpm = np.max(hpm_5min)\n",
    "            hpm_gain = max_hpm - np.mean(hpm[:5])\n",
    "            if experiment_type == 'IT':\n",
    "                shallow_thresh = 250\n",
    "                deep_thresh = 350\n",
    "                if e2_depths < shallow_thresh:\n",
    "                    vals.append(hpm_gain)\n",
    "                    ids.append(\"IT Shallow\")\n",
    "                    num_itshallow += 1\n",
    "                elif e2_depths > deep_thresh:\n",
    "                    vals.append(hpm_gain)\n",
    "                    ids.append(\"IT Deep\")\n",
    "                    num_itdeep += 1\n",
    "            else:\n",
    "                vals.append(hpm_gain)\n",
    "                ids.append(\"PT\")\n",
    "                num_pt += 1 \n",
    "\n",
    "    # Plot some rectangles\n",
    "    df = pd.DataFrame({\n",
    "        'Values': vals, \"Neuron Identity\": ids\n",
    "        })\n",
    "    sns.barplot(\n",
    "        x='Neuron Identity', y='Values', data=df\n",
    "        )\n",
    "    plt.title('Gain in HPM from Experiment Beginning')\n",
    "    plt.show(block=True)\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 121,
   "metadata": {
    "collapsed": true
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plot_itpt_hpm()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Analysing all times/sessions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {
    "code_folding": [
     0
    ]
   },
   "outputs": [],
   "source": [
    "def plot_itpt_hpm(bin_size=1, plotting_bin_size=10, num_minutes=200, first_N_experiments=5):\n",
    "    \"\"\"\n",
    "    Aggregates hits per minute across all IT and PT animals. Performs regression\n",
    "    on the resulting data, and returns the p-value of how different linear\n",
    "    regression between the two animals are.\n",
    "    \"\"\"\n",
    "\n",
    "    # Getting all hits per minute arrays\n",
    "    ITshallow_train = []\n",
    "    ITshallow_target = []\n",
    "    ITdeep_train = []\n",
    "    ITdeep_target = []\n",
    "    PT_train = []\n",
    "    PT_target = []\n",
    "    num_itshallow = 0\n",
    "    num_itdeep = 0\n",
    "    num_pt = 0\n",
    "    \n",
    "    for animaldir in os.listdir(datadir):\n",
    "        animal_path = datadir + animaldir + '/'\n",
    "        if not os.path.isdir(animal_path):\n",
    "            continue\n",
    "        animal_path_files = os.listdir(animal_path)\n",
    "        animal_path_files.sort()\n",
    "        animal_path_files = animal_path_files[:first_N_experiments]\n",
    "        for file_name in animal_path_files:\n",
    "            result = re.search(pattern, file_name)\n",
    "            if not result:\n",
    "                continue\n",
    "            experiment_type = result.group(1)\n",
    "            experiment_animal = result.group(2)\n",
    "            experiment_date = result.group(3)\n",
    "            f = h5py.File(animal_path + file_name, 'r')\n",
    "            com_cm = np.array(f['com_cm'])\n",
    "            try:\n",
    "                e2_indices = e2_dict[animaldir][file_name]\n",
    "            except:\n",
    "                continue\n",
    "            ens_neur = np.array(f['ens_neur'])\n",
    "            e2_neur = ens_neur[e2_indices]\n",
    "            e2_depths = np.mean(com_cm[e2_neur,2])\n",
    "            xs, hpm, _, _ =\\\n",
    "                learning_params(\n",
    "                    experiment_type + experiment_animal,\n",
    "                    experiment_date,\n",
    "                    bin_size=bin_size\n",
    "                    )\n",
    "            xs = xs*bin_size\n",
    "            if experiment_type == 'IT':\n",
    "                shallow_thresh = 250\n",
    "                deep_thresh = 350\n",
    "                for idx, x_val in enumerate(xs):\n",
    "                    if x_val <= num_minutes:\n",
    "                        if e2_depths < shallow_thresh:\n",
    "                            ITshallow_train.append(x_val)\n",
    "                            ITshallow_target.append(hpm[idx])\n",
    "                        elif e2_depths > deep_thresh:\n",
    "                            ITdeep_train.append(x_val)\n",
    "                            ITdeep_target.append(hpm[idx])\n",
    "                if e2_depths < shallow_thresh:\n",
    "                    num_itshallow += 1\n",
    "                elif e2_depths > deep_thresh:\n",
    "                    num_itdeep += 1\n",
    "            else:\n",
    "                for idx, x_val in enumerate(xs):\n",
    "                    if x_val <= num_minutes:\n",
    "                        PT_train.append(x_val)\n",
    "                        PT_target.append(hpm[idx])\n",
    "                num_pt += 1\n",
    "\n",
    "    # Collect data\n",
    "    ITshallow_train = np.array(ITshallow_train).squeeze()\n",
    "    ITshallow_target = np.array(ITshallow_target)\n",
    "    ITdeep_train = np.array(ITdeep_train).squeeze()\n",
    "    ITdeep_target = np.array(ITdeep_target)\n",
    "    PT_train = np.array(PT_train).squeeze()\n",
    "    PT_target = np.array(PT_target)\n",
    "    fig = plt.figure()\n",
    "    ax = fig.add_subplot(111)\n",
    "\n",
    "    # p-val for linear regression slope similarity\n",
    "    p_val = linreg_pval(\n",
    "        ITshallow_train, ITshallow_target,\n",
    "        ITdeep_train, ITdeep_target\n",
    "    )\n",
    "    print(\"Comparing linear regression slopes of IT and PT:\")\n",
    "    print(\"p-val = \" + str(p_val))\n",
    "\n",
    "    # Some options:\n",
    "    # Order 1, Order 2, Logx True\n",
    "    sns.regplot(\n",
    "        ITshallow_train, ITshallow_target,\n",
    "        x_bins=plotting_bin_size,\n",
    "        color='forestgreen', label='IT shallow (%d Experiments)'%num_itshallow,\n",
    "        fit_reg=False\n",
    "        )\n",
    "    sns.regplot(\n",
    "        ITdeep_train, ITdeep_target,\n",
    "        x_bins=plotting_bin_size,\n",
    "        color='cornflowerblue', label='IT deep (%d Experiments)'%num_itdeep,\n",
    "        fit_reg=False\n",
    "        )\n",
    "    sns.regplot(\n",
    "        PT_train, PT_target,\n",
    "        x_bins=plotting_bin_size,\n",
    "        color='coral', label='PT (%d Experiments)'%num_pt,\n",
    "        fit_reg=False\n",
    "        )\n",
    "    ax.set_ylabel('Number of Hits')\n",
    "    ax.set_xlabel('Minutes into the Experiment')\n",
    "    plt.title('Hits/%d-min of All Experiments'%bin_size)\n",
    "    plt.legend()\n",
    "    plt.show(block=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {
    "code_folding": [
     0
    ],
    "collapsed": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Comparing linear regression slopes of IT and PT:\n",
      "p-val = [0.08503738]\n"
     ]
    },
    {
     "data": {
      "image/png": "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VvZOlcSVV4bmBsQ6msMgWjoijTGmLBYUwsB3qcgYd0qoGGIsI6sxJupYUAgTz1KRjWo3AqBR7Ua2VKQxJupYUAijlMQUUpNTAUhNTrUagjEm6lhQMCZIsnNdfLE2m9lfHeCLtdlk51Z8uHdFUmc//fTTvPLKKxUuQ2mys7MZOHCgd/YuOGkczj333CJj8QcNGkT37t29n8NfOoc5c+Zw1llnefdJT08vkh4iFEaNGhXyc0yfPt2br6gk48ePZ/ny5SEtR6Bs9JExQeBZ5tXlclKzJ8XDrGUHGNajNi2Ordh6EqWlzo60d955h27duhEXF+d97amnnqJDhw6H7fvEE0/4zYfk69JLL2X06NGl7hMsBQUFjBs3LuTnefXVV+nVqxc1avhPhQIwcOBA7r///pAMMS0vqykYU0G+y7x61urIyYfsPNyvh2+C6JQpU+jevTuDBw/mt99+876+ceNGhg4dSt++fbnqqqu86bNLS+s8YsQIrrnmGi666CJmzZrl93wLFy6kS5cu3uerV69m586d3sRwwfDhhx8yePBgXC4X27dvp3v37uzYsYM5c+Zw0003MXToULp3714k19P8+fO54oorSE9PZ/To0d6azOmnn86kSZPo168f3333HYMGDeKHH37wbpswYQJ9+/Zl8ODBrFq1ypvW+uOPPwbKTrE9bNgwLr74YoYPH47L5eLVV19l+/btXHvttQwaNIiCggLuueceevToQc+ePb3J/xo3bsyePXsqtLZysFhNwZgKKm2ZV5cLVqzL5dw2SSEvx+rVq1m8eDHz5s2joKCAPn36cNJJJwFw//3389BDD9GsWTO+//57HnroIV599dVS0zqrKrNmzeLAgQP06dOHzp07e3MlgZOMbdOmTaSlpQFO4rfx48fz+OOP+20Kuffee4mNjeWiiy7i5ptv9rt2wuLFi72BCWDmzJl069aNJUuW8Prrr/PFF19w2223ccwxxwCH0lrXqFGDK664gs6dO1OzZk3ee+893nzzTRISEnjwwQdZuHAhvXv35sCBA7Ro0YLbb7/9sHMfOHCADh06MGLECG655Raeeuoppk6dyvr167n77rvp0qULs2fPLjHF9tq1a3n33Xdp0KABAwYMYOXKlVxzzTVMnz6dGTNmUK9ePVavXs22bdtYtMiZab93717v+du0acO3335L9+7dy/cPH2QWFIypoNKWec3Jh+1/FfjfGGQZGRl07drV20xx4YUXArB//36+++67IhdCT7K50tI6d+nSheTkZJKTkznzzDP54YcfigSF3bt3U7t2be/zN954g/POO49jj/VdO8vxxBNP0LBhQ7Kyshg2bBjz58/3O4GtpOaj+++/nx49etC2bVt69Ojhfb1jx47etNHdunVj5cqVxMfHs3r1aq644grA6feoX78+4GQiLemim5CQwHnnnQc4s4UTExNJSEigZcuWbN68GaDUFNunnnoqjRo5owtbtWrF5s2bD0sK2KRJEzZt2sSYMWPo3LmzNxMrQP369Q/L6BoJFhRMleMZ1RWu0V2lLfOaFA8N6sQdviFE/N19u1wujjrqKL9LW3rSOvtLv1zWKmjF02B/9913rFy5kjfffJP9+/eTl5dHzZo1ufPOO73BpFatWvTo0YNVq1aVa1bztm3biI2N5c8//6SwsNC7YE5JabD79OnD8OHDDztOUlJSkf4PXwkJCd7jlZQG25Ni+9xzzy3y3q+//vqwNNi+ne8ederUYf78+Xz55Ze88cYbvPfee97Egjk5OUFPg30krE/BVDnDOg6jc/PODOs4LCznK22Z15gYaN8i0f/GYJejfXs+/PBDsrOzycrK8i41WatWLdLS0rzNQi6Xi59++gkoPa3zxx9/TE5ODrt37+abb745rJO4Tp06FBQUkJOTAzjLSC5dupRPPvmEu+++m969e3PnnXeSn5/vXSMhLy+PpUuX0qJFi4A/V35+PiNHjmTixImccMIJTJs2zbtt2bJl7Nmzh+zsbD766CPOOOMMzj77bJYsWeId4bRnzx7vnX5FHUmKbd9U2Lt27cLlctG9e3duv/32IunNf//993L9XUKl2tYUkhNjvI8pheG9szSh1alpJzo1DV5HZ1l8l3n1HX0UEwPDetQmOSE8qziddNJJXHrppaSnp9O4cWPvegcAEyZM4MEHH2TKlCnk5+dz6aWX0qpVq1LTOp966qnccMMNbNmyhZtvvrlI05FHp06dWLlyJR07diyxXLm5ufzzn/8kLy+PwsJCzj777BLz9hTvU3jggQf46quvaNeuHe3ataNVq1ZcccUVnH/++QD8/e9/56677mLDhg307NnTG7juuOMOhgwZQmFhIQkJCYwePToo6w4cSYrtK6+8kuuvv55jjjmGUaNGMXLkSAoLneVb//Of/wBOsNywYQMnn3xyhctYUSFNnS0iFwOTcNZofllV/WajEpErgLeB9qpa6qozwUqd/WNmHh/8L5uL2iazq+AbXsl4haHthob8YhKstM4m9MqbOjs7z8WKdbls/6uABnXiaN8iMWwBIdiefvppatasydChQ0vdb+3atUybNs27NGc4zZkzh9WrV4dtCGsoffjhh6xZs4Y77rgjKMeLytTZIhIHPAt0AzKBFSKyQFXXFtuvNjAM+DpUZfGndVoCrdM848fDe2dpqqbkhJiwjDKKJm3atOHMM8+koKCgxLZ6U7b8/HyGDBkS6WIAoW0+6gD8oqq/AojIW0A6UHyNwDHA48CdISyLMaYcbrvttoD39YzyCbe+ffvSt2/fiJw72C655JJIF8ErlB3NjYFNPs8z3a95icjpQBNVtXYUY4yJAqGsKfhrTPV2YIhILPAkMDiEZTDGGFMOoawpZAJNfJ6nAX/4PK8NnAwsFZHfgbOABSJiuaSNMSZCQllTWAG0EJHmwGagP3CVZ6Oq/gV4F5YVkaXAnWWNPjLGGBM6IQsKqpovIrcCS3CGpE5V1TUi8jCQoaoLQnVuYyIi5yCs/hJ2bYF6x8LJ50BSyZkxA9G6dWtatmxJQUEBxx9/PKNGjeKGG24A4M8//yQ2NpZ69eoB8PbbbxeZVetyubj22mt57rnnqFWrFiNHjmTp0qXUr1/fm3sHnLTNn376KQkJCRx33HE8+uijHHXUUSxbtoyJEyeSl5dHQkICI0aM8JvFc9CgQWzfvt07G7dp06ZMnjy5Qp+7NNu2bWPcuHEhPUdmZibfffcdPXv2LHGf3NxcrrvuOmbMmEF8fNWZ8hXST6Kqi4HFxV7zO6hYVc8PZVmMCakNa+H1seAqhLwcSEiCJdPg6vugaZsjPmxycrI3PcXw4cNZvHix93lZcwk+++wzWrVqRa1atQBntM7AgQO5++67i+zXqVMnhg8fTnx8PBMmTOCFF15gxIgR1K1blylTptCwYUN+/vlnhg4dyhdffOH3XIGkxQ6G/Px8GjZsGNKAALB582YWLVpUalBITEzk7LPPZvHixfTq1Suk5QmnqhPejImUnINOQMj1WUglz0n9wOtjYfgrFa4xALRr1w5VDXj/hQsXFpk53L59ezIzMw/bzzcpW9u2bXn//fcBZw6CR4sWLcjNzSU3N7dIbaQ0N910E927d6d379689dZbrFixgokTJzJo0CBatWrFDz/8QFZWFo888ginnnoqBw4cYMyYMfz8888UFBRw66230rVrV+bMmcPSpUvJzc3lwIEDPPLII9x4440sWrSIOXPm8NFHH1FYWMjPP//MkCFDyMvLY/78+SQmJvLiiy+SmprKxo0beeihh9i9ezfJycmMGTOGE044gXvuuYdatWqxevVqduzYwYgRI7j44ouZOHEi69evJz09nT59+tCpUydGjhzpnZX99NNP06xZM7p27crEiRMtKBhjfKz+0qkh+OMqhDXL4IyuFTpFfn4+n3/++WGJ2Erz7bff8tBDD5XrPO+8847fMfNLliyhdevWJQaEO++809t81LFjR+6++27GjBnDgAEDSEtLY9q0acycOdO7/8GDB72B4t5772XRokU8//zznHXWWTz66KPs3buXfv36edNn/O9//2PBggWkpqYeFtjWrVvH3Llzyc3NpVu3btx5553MmzePRx55hHnz5jF48OASU4cDbN++nTfeeINff/2Vm266ybsewtSpU3nhhRcAGDNmDNdccw29evUiNzfXm6aiRYsW3vUYqgoLCsZU1K4th2oGxeXlwM4tR3zo7Oxs0tPTAaemUJ6JYnv27PE2HQViypQpxMXFHXbXu27dOp544gmmTp1a4nv9NR8dffTRDBs2jGuuuYZnnnmG1NRU77bLLrsMcGovWVlZ7N27ly+//JJPPvnEe56cnBy2bHH+dp06dSryfl9nnnmm93PWrl3bmzK8ZcuWqGqpqcMBunbtSmxsLCeeeCJ//vmn33O0bduW559/nq1bt3LRRRfRrFkzwMmGmpCQQFZWVrn+1tHMgoIxFVXvWKcPwV9gSEiC+oevLxAo3z6F8oqPjy+SZro0c+fOZenSpUyfPr1IOuqtW7dy6623Mn78eI477rhyl+Hnn38mNTX1sHUC/KW8Bpg8eTLHH398kW3ff/99qUtZ+tZeYmNjSUhI8P5eUFBQaurw4u8vSc+ePTnttNNYunQpQ4cOZezYsd5O99zcXJKSqk56E0udbUxFnXwOxJTwXykmFk6KTF6t5s2bs2nTpjL3+/zzz3nppZeYMmVKkYvv3r17ueGGG/jPf/5TJONqoFatWsXnn3/O3LlzmTp1apGyLF7sjD/JyMigdu3a1K5d25vG25Ok0zetdEWUljq8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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plot_itpt_hpm(\n",
    "    bin_size=1, plotting_bin_size=10, num_minutes=100,\n",
    "    first_N_experiments=20\n",
    "    )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 194,
   "metadata": {
    "code_folding": [
     0
    ]
   },
   "outputs": [],
   "source": [
    "def plot_itpt_hpm(bin_size=1, plotting_bin_size=10, num_minutes=200, first_N_experiments=5):\n",
    "    \"\"\"\n",
    "    Aggregates hits per minute across all IT and PT animals. Performs regression\n",
    "    on the resulting data, and returns the p-value of how different linear\n",
    "    regression between the two animals are.\n",
    "    \"\"\"\n",
    "\n",
    "    # Getting all hits per minute arrays\n",
    "    ITshallow_train = []\n",
    "    ITshallow_target = []\n",
    "    ITdeep_train = []\n",
    "    ITdeep_target = []\n",
    "    PT_train = []\n",
    "    PT_target = []\n",
    "    num_itshallow = 0\n",
    "    num_itdeep = 0\n",
    "    num_pt = 0\n",
    "    \n",
    "    for animaldir in os.listdir(datadir):\n",
    "        animal_path = datadir + animaldir + '/'\n",
    "        if not os.path.isdir(animal_path):\n",
    "            continue\n",
    "        animal_path_files = os.listdir(animal_path)\n",
    "        animal_path_files.sort()\n",
    "        animal_path_files = animal_path_files[:first_N_experiments]\n",
    "        for file_name in animal_path_files:\n",
    "            result = re.search(pattern, file_name)\n",
    "            if not result:\n",
    "                continue\n",
    "            experiment_type = result.group(1)\n",
    "            experiment_animal = result.group(2)\n",
    "            experiment_date = result.group(3)\n",
    "            f = h5py.File(animal_path + file_name, 'r')\n",
    "            com_cm = np.array(f['com_cm'])\n",
    "            try:\n",
    "                e2_indices = e2_dict[animaldir][file_name]\n",
    "            except:\n",
    "                continue\n",
    "            ens_neur = np.array(f['ens_neur'])\n",
    "            e2_neur = ens_neur[e2_indices]\n",
    "            e2_depths = np.mean(com_cm[e2_neur,2])\n",
    "            try:\n",
    "                xs, hpm, _, _ =\\\n",
    "                    learning_params(\n",
    "                        experiment_type + experiment_animal,\n",
    "                        experiment_date,\n",
    "                        bin_size=1\n",
    "                        )\n",
    "            except:\n",
    "                print(\"Binning error with \" + f.filename)\n",
    "            hpm = np.convolve(hpm, np.ones((3,))/3)\n",
    "            if experiment_type == 'IT':\n",
    "                shallow_thresh = 250\n",
    "                deep_thresh = 350\n",
    "                for idx, x_val in enumerate(xs):\n",
    "                    if x_val <= num_minutes:\n",
    "                        if e2_depths < shallow_thresh:\n",
    "                            ITshallow_train.append(x_val)\n",
    "                            ITshallow_target.append(hpm[idx])\n",
    "                        elif e2_depths > deep_thresh:\n",
    "                            ITdeep_train.append(x_val)\n",
    "                            ITdeep_target.append(hpm[idx])\n",
    "                if e2_depths < shallow_thresh:\n",
    "                    num_itshallow += 1\n",
    "                elif e2_depths > deep_thresh:\n",
    "                    num_itdeep += 1\n",
    "            else:\n",
    "                for idx, x_val in enumerate(xs):\n",
    "                    if x_val <= num_minutes:\n",
    "                        PT_train.append(x_val)\n",
    "                        PT_target.append(hpm[idx])\n",
    "                num_pt += 1\n",
    "\n",
    "    # Collect data\n",
    "    ITshallow_train = np.array(ITshallow_train).squeeze()\n",
    "    ITshallow_target = np.array(ITshallow_target)\n",
    "    ITdeep_train = np.array(ITdeep_train).squeeze()\n",
    "    ITdeep_target = np.array(ITdeep_target)\n",
    "    PT_train = np.array(PT_train).squeeze()\n",
    "    PT_target = np.array(PT_target)\n",
    "    fig = plt.figure()\n",
    "    ax = fig.add_subplot(111)\n",
    "    \n",
    "    # Simpl plot\n",
    "    sns.pointplot(\n",
    "        ITshallow_train, ITshallow_target,\n",
    "        color='forestgreen', label='IT shallow (%d Experiments)'%num_itshallow\n",
    "        )\n",
    "    sns.pointplot(\n",
    "        ITdeep_train, ITdeep_target,\n",
    "        color='cornflowerblue', label='IT deep (%d Experiments)'%num_itdeep\n",
    "        )\n",
    "    sns.pointplot(\n",
    "        PT_train, PT_target,\n",
    "        color='coral', label='PT (%d Experiments)'%num_pt\n",
    "        )\n",
    "    ax.set_ylabel('Number of Hits')\n",
    "    ax.set_xlabel('Minutes into the Experiment')\n",
    "    plt.title('Hits/min of All Experiments')\n",
    "    plt.xticks(np.arange(0,65,5), np.arange(0,65,5))\n",
    "    plt.legend()\n",
    "    plt.savefig('sfn_fig3.eps')\n",
    "    plt.show(block=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 195,
   "metadata": {
    "code_folding": [],
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "No handles with labels found to put in legend.\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plot_itpt_hpm(\n",
    "    bin_size=5, plotting_bin_size=10, num_minutes=100,\n",
    "    first_N_experiments=20\n",
    "    )"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Analysing all sessions, in the first 10 minutes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 196,
   "metadata": {
    "code_folding": []
   },
   "outputs": [],
   "source": [
    "def plot_itpt_hpm(bin_size=1, plotting_bin_size=10, num_minutes=200, first_N_experiments=5):\n",
    "    \"\"\"\n",
    "    Aggregates hits per minute across all IT and PT animals. Performs regression\n",
    "    on the resulting data, and returns the p-value of how different linear\n",
    "    regression between the two animals are.\n",
    "    \"\"\"\n",
    "\n",
    "    # Getting all hits per minute arrays\n",
    "    ITshallow_train = []\n",
    "    ITshallow_target = []\n",
    "    ITdeep_train = []\n",
    "    ITdeep_target = []\n",
    "    PT_train = []\n",
    "    PT_target = []\n",
    "    num_itshallow = 0\n",
    "    num_itdeep = 0\n",
    "    num_pt = 0\n",
    "    \n",
    "    for animaldir in os.listdir(datadir):\n",
    "        animal_path = datadir + animaldir + '/'\n",
    "        if not os.path.isdir(animal_path):\n",
    "            continue\n",
    "        animal_path_files = os.listdir(animal_path)\n",
    "        animal_path_files.sort()\n",
    "        animal_path_files = animal_path_files[:first_N_experiments]\n",
    "        for file_name in animal_path_files:\n",
    "            result = re.search(pattern, file_name)\n",
    "            if not result:\n",
    "                continue\n",
    "            experiment_type = result.group(1)\n",
    "            experiment_animal = result.group(2)\n",
    "            experiment_date = result.group(3)\n",
    "            f = h5py.File(animal_path + file_name, 'r')\n",
    "            com_cm = np.array(f['com_cm'])\n",
    "            try:\n",
    "                e2_indices = e2_dict[animaldir][file_name]\n",
    "            except:\n",
    "                continue\n",
    "            ens_neur = np.array(f['ens_neur'])\n",
    "            e2_neur = ens_neur[e2_indices]\n",
    "            e2_depths = np.mean(com_cm[e2_neur,2])\n",
    "            xs, hpm, _, _ =\\\n",
    "                learning_params(\n",
    "                    experiment_type + experiment_animal,\n",
    "                    experiment_date,\n",
    "                    bin_size=bin_size\n",
    "                    )\n",
    "            xs = xs*bin_size\n",
    "            if experiment_type == 'IT':\n",
    "                shallow_thresh = 250\n",
    "                deep_thresh = 350\n",
    "                for idx, x_val in enumerate(xs):\n",
    "                    if x_val <= num_minutes:\n",
    "                        if e2_depths < shallow_thresh:\n",
    "                            ITshallow_train.append(x_val)\n",
    "                            ITshallow_target.append(hpm[idx])\n",
    "                        elif e2_depths > deep_thresh:\n",
    "                            ITdeep_train.append(x_val)\n",
    "                            ITdeep_target.append(hpm[idx])\n",
    "                if e2_depths < shallow_thresh:\n",
    "                    num_itshallow += 1\n",
    "                elif e2_depths > deep_thresh:\n",
    "                    num_itdeep += 1\n",
    "            else:\n",
    "                for idx, x_val in enumerate(xs):\n",
    "                    if x_val <= num_minutes:\n",
    "                        PT_train.append(x_val)\n",
    "                        PT_target.append(hpm[idx])\n",
    "                num_pt += 1\n",
    "\n",
    "    # Collect data\n",
    "    ITshallow_train = np.array(ITshallow_train).squeeze()\n",
    "    ITshallow_target = np.array(ITshallow_target)\n",
    "    ITdeep_train = np.array(ITdeep_train).squeeze()\n",
    "    ITdeep_target = np.array(ITdeep_target)\n",
    "    PT_train = np.array(PT_train).squeeze()\n",
    "    PT_target = np.array(PT_target)\n",
    "    fig = plt.figure()\n",
    "    ax = fig.add_subplot(111)\n",
    "\n",
    "    # p-val for linear regression slope similarity\n",
    "    p_val = linreg_pval(\n",
    "        ITshallow_train, ITshallow_target,\n",
    "        ITdeep_train, ITdeep_target\n",
    "    )\n",
    "    print(\"Comparing linear regression slopes of IT and PT:\")\n",
    "    print(\"p-val = \" + str(p_val))\n",
    "\n",
    "    # Some options:\n",
    "    # Order 1, Order 2, Logx True\n",
    "    sns.regplot(\n",
    "        ITshallow_train, ITshallow_target,\n",
    "        x_bins=plotting_bin_size,\n",
    "        color='forestgreen', label='IT shallow (%d Experiments)'%num_itshallow\n",
    "        )\n",
    "    sns.regplot(\n",
    "        ITdeep_train, ITdeep_target,\n",
    "        x_bins=plotting_bin_size,\n",
    "        color='cornflowerblue', label='IT deep (%d Experiments)'%num_itdeep\n",
    "        )\n",
    "    sns.regplot(\n",
    "        PT_train, PT_target,\n",
    "        x_bins=plotting_bin_size,\n",
    "        color='coral', label='PT (%d Experiments)'%num_pt\n",
    "        )\n",
    "    ax.set_ylabel('Number of Hits')\n",
    "    ax.set_xlabel('Minutes into the Experiment')\n",
    "    plt.title('Hits/%d-min of All Experiments'%bin_size)\n",
    "    plt.legend()\n",
    "    from matplotlib.collections import PolyCollection as p\n",
    "    for c in ax.findobj(p):\n",
    "        c.set_zorder(-1)\n",
    "        c.set_rasterized(True)\n",
    "    #everything on zorder -1 or lower will be rasterized\n",
    "    ax.set_rasterization_zorder(0)\n",
    "    plt.savefig('sfn_fig4b.eps')\n",
    "    plt.show(block=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 197,
   "metadata": {
    "code_folding": [],
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Comparing linear regression slopes of IT and PT:\n",
      "p-val = [0.01440257]\n"
     ]
    },
    {
     "data": {
      "image/png": "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1FEsXryYSCTCH/7wB4YNG8bevXv56U9/yo4dOwC47bbbOOGEE7j//vvZsWMH27dvZ8eOHVx11VVceeWV3HfffWzdupXzzz+fU089tdnfRnGQIhWKLjBhwgS5bt06ef3118twOCzPO+88+e6778rrrrtOSinl0qVL5Z133imllPKWW26R119/vXQcR65fv17OmDFDSinTjpdSyjvvvFMuXbpUSinl1KlTZWVlpZRSysrKSvmNb3xD1tXVSSmlfOihh+T9998vq6qq5Jlnnild15VSSlldXd3Mzq7Y8ec//1neeuutUkopN2zYIKdMmSLD4XDadZues3TpUjlt2jS5f/9+GQ6H5RlnnCF37NjRqu1NueWWW+TUqVPleeedJ8877zx50003SSmlrK+vl2effbZcuXKlPPPMM+WWLVuSx19zzTXScRy5efNmOXnyZBkOh+Xjjz8u//CHP0gppYxEIvLCCy+UW7dule+++64cP3683Lp1a9rfMPEsJ5xwgqyoqJCRSEROmjRJLly4UEop5aOPPip//vOfSymlvOmmm+T7778vpZSyrKxMzp49W0op5e9+9zv59a9/XUYiEVlZWSlPPvlkGY1G5bZt2+ScOXOaPavi4EaNFBRd5uijj2b79u08//zz7abTmDFjBpqmceSRR7Jnz55O3Wf16tVs2LCByy+/HPAS/02YMIGsrCx8Ph+33347Z5xxBmeccUa71+qIHR9++CHOze5+AAADX0lEQVRXXHEFAEcccQSHHXYYmzdv5uijj27z2qeccgrZ2dnJ88rKyqipqWnR9paYN28es2fPTtsWCAT42c9+xhVXXMH8+fMZNmxYct9ZZ52FpmmMGDGCkpISNm3axNtvv80XX3zBiy++CEBNTQ1btmzBNE3GjRtHSUlJi/ceN25cMjnksGHDOO200wA46qijWLVqFeAlukst3FRbW0ttbS0AU6ZMwbIs8vPzyc/Pp7Kyss3fleLgRYmC4oCYNm0a9957L4sXL2bfvn2tHmdZVrNtuq7jum7yc2slG6WUnHbaafz6179utu+pp55i5cqVLF++nMcee4zFixe3aW9LdrR0v66Qem1d13Ecp03bO8qXX35Jbm5uMitngqZp34UQSCn58Y9/zOTJk9P2rVq1imAw2CHbNU1LftY0LTkH4bouTzzxBH6/v83zdV3Htu0OPp3iYENNNCsOiEsuuYTvfe97jB49utPnDhkyhI0bNxKNRqmpqWHlypXJfaFQiLq6OgAmTJjARx99xJYtWwBoaGhg8+bN1NXVUVNTw5QpU7jtttv4/PPPu/QMqfcCr9JWIrJo8+bN7Ny5k8MPP7zNc1qjNds7SllZGYsWLeKZZ57hjTfeYPXq1cl9L7zwAq7rsnXrVrZt28bIkSOZNGkSf/vb34jFYkn7u6twz6RJk3jssceSn9etW9fm8R39HSkOLtRIQXFAFBcXc9VVV3Xp3MGDBzN79mzOPfdcRowYwTHHHJPcd+mll/Jv//ZvDBo0iCVLlnDPPfdw0003EY1GAbjxxhsJhUJ873vfS44w5s+f3yU7Ro8eja7rnHfeeVx00UV84xvf4Kc//Snnnnsuuq5zzz33NBthND2ntTKa+fn5Ldo+cuTIZsfee++9PPDAA8nPTz75JLfffjvz5s2jqKiIu+66i/nz5/PUU08BMHLkSK644goqKyu588478fl8fO1rX6OsrIyLLroIKSV5eXn88Y9/7NLvpSm33347CxYs4Nxzz8VxHE488cQ2w3Dz8vI4/vjjOeecc5g8ebKaaO4jqCypCkUf5NZbb+WMM85oNgehUBwoyn2kUCgUiiRqpKBQKBSKJGqkoFAoFIokShQUCoVCkUSJgkKhUCiSKFFQKBQKRRIlCgqFQqFI8v8BTFKLHRlDRWYAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plot_itpt_hpm(\n",
    "    bin_size=1, plotting_bin_size=10, num_minutes=10,\n",
    "    first_N_experiments=20\n",
    "    )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 178,
   "metadata": {
    "code_folding": [
     0
    ]
   },
   "outputs": [],
   "source": [
    "def plot_itpt_hpm(bin_size=1, plotting_bin_size=10, num_minutes=200, first_N_experiments=5):\n",
    "    \"\"\"\n",
    "    Aggregates hits per minute across all IT and PT animals. Performs regression\n",
    "    on the resulting data, and returns the p-value of how different linear\n",
    "    regression between the two animals are.\n",
    "    \"\"\"\n",
    "\n",
    "    # Getting all hits per minute arrays\n",
    "    ITshallow_train = []\n",
    "    ITshallow_target = []\n",
    "    ITdeep_train = []\n",
    "    ITdeep_target = []\n",
    "    PT_train = []\n",
    "    PT_target = []\n",
    "    num_itshallow = 0\n",
    "    num_itdeep = 0\n",
    "    num_pt = 0\n",
    "    \n",
    "    for animaldir in os.listdir(datadir):\n",
    "        animal_path = datadir + animaldir + '/'\n",
    "        if not os.path.isdir(animal_path):\n",
    "            continue\n",
    "        animal_path_files = os.listdir(animal_path)\n",
    "        animal_path_files.sort()\n",
    "        animal_path_files = animal_path_files[:first_N_experiments]\n",
    "        for file_name in animal_path_files:\n",
    "            result = re.search(pattern, file_name)\n",
    "            if not result:\n",
    "                continue\n",
    "            experiment_type = result.group(1)\n",
    "            experiment_animal = result.group(2)\n",
    "            experiment_date = result.group(3)\n",
    "            f = h5py.File(animal_path + file_name, 'r')\n",
    "            com_cm = np.array(f['com_cm'])\n",
    "            try:\n",
    "                e2_indices = e2_dict[animaldir][file_name]\n",
    "            except:\n",
    "                continue\n",
    "            ens_neur = np.array(f['ens_neur'])\n",
    "            e2_neur = ens_neur[e2_indices]\n",
    "            e2_depths = np.mean(com_cm[e2_neur,2])\n",
    "            xs, hpm, _, _ =\\\n",
    "                learning_params(\n",
    "                    experiment_type + experiment_animal,\n",
    "                    experiment_date,\n",
    "                    bin_size=1\n",
    "                    )\n",
    "            hpm = np.convolve(hpm, np.ones((bin_size,))/bin_size)\n",
    "            if experiment_type == 'IT':\n",
    "                shallow_thresh = 250\n",
    "                deep_thresh = 350\n",
    "                for idx, x_val in enumerate(xs):\n",
    "                    x_val = int(x_val)\n",
    "                    if x_val <= num_minutes:\n",
    "                        if e2_depths < shallow_thresh:\n",
    "                            ITshallow_train.append(x_val)\n",
    "                            ITshallow_target.append(hpm[idx])\n",
    "                        elif e2_depths > deep_thresh:\n",
    "                            ITdeep_train.append(x_val)\n",
    "                            ITdeep_target.append(hpm[idx])\n",
    "                if e2_depths < shallow_thresh:\n",
    "                    num_itshallow += 1\n",
    "                elif e2_depths > deep_thresh:\n",
    "                    num_itdeep += 1\n",
    "            else:\n",
    "                for idx, x_val in enumerate(xs):\n",
    "                    if x_val <= num_minutes:\n",
    "                        PT_train.append(x_val)\n",
    "                        PT_target.append(hpm[idx])\n",
    "                num_pt += 1\n",
    "\n",
    "    # Collect data\n",
    "    ITshallow_train = np.array(ITshallow_train).squeeze()\n",
    "    ITshallow_target = np.array(ITshallow_target)\n",
    "    ITdeep_train = np.array(ITdeep_train).squeeze()\n",
    "    ITdeep_target = np.array(ITdeep_target)\n",
    "    PT_train = np.array(PT_train).squeeze()\n",
    "    PT_target = np.array(PT_target)\n",
    "    fig = plt.figure()\n",
    "    ax = fig.add_subplot(111)\n",
    "\n",
    "    # p-val for linear regression slope similarity\n",
    "    p_val = linreg_pval(\n",
    "        ITshallow_train, ITshallow_target,\n",
    "        ITdeep_train, ITdeep_target\n",
    "    )\n",
    "    print(\"Comparing linear regression slopes of IT and PT:\")\n",
    "    print(\"p-val = \" + str(p_val))\n",
    "\n",
    "    # Some options:\n",
    "    # Order 1, Order 2, Logx True\n",
    "    sns.pointplot(\n",
    "        ITshallow_train, ITshallow_target,\n",
    "        color='forestgreen', label='IT shallow (%d Experiments)'%num_itshallow\n",
    "        )\n",
    "    sns.pointplot(\n",
    "        ITdeep_train, ITdeep_target,\n",
    "        color='cornflowerblue', label='IT deep (%d Experiments)'%num_itdeep\n",
    "        )\n",
    "    sns.pointplot(\n",
    "        PT_train, PT_target,\n",
    "        color='coral', label='PT (%d Experiments)'%num_pt\n",
    "        )\n",
    "    ax.set_ylabel('Number of Hits')\n",
    "    ax.set_xlabel('Minutes into the Experiment')\n",
    "    plt.xticks(np.arange(0,10,2), np.arange(0,10,2))\n",
    "    plt.title('Hits/%d-min of All Experiments'%bin_size)\n",
    "    plt.legend()\n",
    "    plt.savefig('sfn_fig3.eps')\n",
    "    plt.show(block=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 179,
   "metadata": {
    "code_folding": [
     0
    ],
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Comparing linear regression slopes of IT and PT:\n",
      "p-val = [0.01053257]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "No handles with labels found to put in legend.\n"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAYUAAAEWCAYAAACJ0YulAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4zLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvnQurowAAIABJREFUeJzs3Xd4FFXbwOHfbE0vhBRK6EgvQaoCkaYUqYKiiAURC2B7EZVX0RcV24dYsAGK0qRLFUFqkA4BghRBYiAEkkASkmzKtpnvj0k2LElIwGzqua8r1+z0kzbPzJlzniMpiqIgCIIgCICmrAsgCIIglB8iKAiCIAgOIigIgiAIDiIoCIIgCA4iKAiCIAgOIigIgiAIDiIoCPkMGDCA/fv3l3UxijR16lS++uqrEj+uoii88cYbdOjQgeHDh9/y/hcvXqRJkybYbDYARo8ezfLly0u6mCWuovzeBdcSQaGK6dmzJ3v27HFatmrVKh5++GHH/IYNG+jUqRMAX375JZMmTbqlc7z11lssXbqUffv2MXDgQNq3b0+nTp0YP348CQkJ//6byDFt2jTGjx9fYsfLdfjwYXbv3s3OnTtZsWJFodvt37+fJk2aMGfOnNs+15dffkmLFi0ICwtzfLVv3/62j/dvXP97L00VJWhWFSIoCCVu165dhIeH06hRI+bOncuhQ4fYtWsXdevW5e233y7r4hUpLi6OWrVq4eHhcdPtVq9ejZ+fH6tXr/5X5+vXrx9HjhxxfB06dOhfHe9W5T7RCAKIoCAUIPdpIiIigu+++46NGzcSFhbGoEGDAPXJolevXoSFhdGzZ0/Wrl3r2Pf06dN4e3sTEhJC9erVCQ4OdqzTarVcuHCh0POuWrWKkSNHMn36dNq3b0+vXr2IjIxk1apVhIeH06VLF3755RfH9q+//jozZ84E1Lv27t2788MPP9ClSxe6du3KypUrCz1XQkICzz77LB07dqRPnz4sW7YMgOXLl/Pmm29y9OhRwsLC+OKLLwrcPysri99++42pU6dy/vx5jh8/Xoyf7K2JjIykU6dOXL58GVB/tu3bt+fcuXOA+nv67rvv6N+/Px06dOCNN97AbDY79t++fTuDBw+mffv2jBw5ktOnTzvW9ezZk9mzZzNw4EDatm2LzWZzeor88ssveeGFF5g0aRJhYWEMHDiQf/75h++++44uXboQHh7OH3/84Theeno6U6ZMoWvXrnTr1o2ZM2dit9uBvCfRjz76iA4dOtCzZ0927twJwMyZMzl06BDTpk0jLCyMadOmoSgK06dPp0uXLtx5550MHDiQM2fOlPjPVyiYrqwLIJRf3bt355lnnuH8+fP83//9HwCZmZm89957rFixggYNGpCYmEhqaqpjn4iICO655x7H/KVLlxg0aBAmkwmtVsu7775703NGRUUxYsQI9u/fzxdffMErr7xCjx49+P333zlw4AATJ07k3nvvxdPTM9++V69eJT09nYiICPbs2cMLL7xA79698fX1zbftf/7zHxo1asSuXbuIjo7mySefJDQ0lBEjRqDValm+fDk///xzoeXctGkTnp6e9O3bl3Xr1rFmzRpatWpV1I/0lrRr146RI0fy2muvMXv2bF599VVeeuklGjZs6Nhm3bp1fP/997i7u/Pss8/y9ddf8/LLL3PixAmmTJnCt99+S8uWLVm7di3PP/88v/32GwaDAVCri2bPno2/vz86Xf5Lwfbt2/n666/58MMPmTJlCk899RQjRowgIiKCX375halTp7Jt2zYAXnvtNapXr87mzZvJysrimWeeoUaNGowcORJQf69Dhw5l3759LF26lP/+97/s2rWLl19+mcjISAYNGsSIESMA9Unz0KFDbNq0CW9vb6Kjo/H29i7Rn61QOPGkUAWNHz+e9u3bO77+97//3dL+Go2Gs2fPkp2dTVBQEI0bN3as27FjB+Hh4Y75mjVrcujQIfbt28eLL75IgwYNbnrs2rVr88ADD6DVaunfvz+XL19m/PjxGAwGunbtisFgKPRpQ6fTMX78ePR6PeHh4Xh4ePDPP//k2+7y5cscPnyYSZMmYTQaadasGSNGjGDNmjXF/hmsXr2afv36odVquf/++1m/fj1Wq7XY+1/vt99+c/p9jB492rFuwoQJmEwmRowYQVBQEKNGjXLad9SoUdSoUQM/Pz+ee+45NmzYAMCyZct46KGHaNOmDVqtlqFDh6LX6zl69Khj39GjR1OjRg3c3NwKLFf79u3p1q0bOp2Ovn37kpKSwrhx49Dr9fTv35+4uDjS0tK4evUqERERTJkyBQ8PDwICAnjiiSccZQH17+DBBx90lOXKlStcvXq1wPPqdDoyMjKIjo5GURQaNmxIUFDQbf1shVsngkIV9NVXX3Ho0CHH163U83t4eDBz5kyWLFlC165dGTdunKM6Iy0tjejoaMLCwvLt5+fnx9ChQ3n++eex2WwcOnTI8WJ1wIABju0CAgIcn3MvVtWrV3csMxqNZGRkFFg2Pz8/pzted3d3MjMz822XmJiIr68vXl5ejmU1a9Ys9kvwy5cvs3//fgYOHAhAr169MJvNjiqRW9W3b1+n38eCBQsc6/R6PUOHDuXMmTOMGTMGSZKc9q1Ro4bT95CYmAioT2jz5s1zCjbx8fGO9TfuW5Abfxf+/v5otVrHPKhPjpcuXcJms9G1a1fHuaZOnUpycrJj/+t/h+7u7o59C9KlSxdGjRrFtGnTuOuuu3jrrbcwmUw3LatQckT1kXBTN16EALp160a3bt3Izs7ms88+46233mLx4sX88ccfdOnSxXHhuJHdbicpKQmTyUT79u05cuSIq4tfoKCgIFJTUzGZTI7AcPnyZaf3HzezZs0aZFnmueeecyyzWCysXr2a3r17l2hZExISmDVrFsOGDePDDz9k5cqVjuqf3HLnunTpkuOOukaNGjz77LNOZbxRQb/b2xESEoLBYGDfvn0FVkPdjscee4zHHnuMpKQkXnrpJebOnctLL71UIscWbk48KQg3FRAQQFxcHLIsA2q9/datW8nMzMRgMODh4eEIAjdWHW3evJno6GhkWSY5OZkPPviA5s2b4+fnVybfS64aNWoQFhbGp59+itls5vTp06xYscJx51+U1atXM2HCBFavXu34+uKLL9ixYwcpKSklVk5FUXj99dcZPnw406dPJygoiM8++8xpm8WLFxMfH8+1a9ccL50BRowYwZIlSzh27BiKopCZmcmOHTtccscdFBTE3XffzYcffojJZEKWZS5cuMCBAweKtX/16tWJjY11zEdFRXHs2DGsVivu7u4YDIZCbzSEkieCgnBTffv2BaBTp04MHToUWZaZN28e3bp1o2PHjhw8eJC3334bRVHYs2cP3bp1c+ybkJDA2LFjadeuHQMHDkSj0TBr1qyy+lacfPrpp8TFxdGtWzcmTJjAxIkTufvuu4vc7+jRo8TFxTFq1CgCAwMdX7169aJu3bpO9ejFldu66/qvpKQk5s+fz9WrV3nxxReRJInp06ezatUqpyar999/P2PGjKF3796EhoY6ngxatWrFu+++y7Rp0+jQoQP3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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plot_itpt_hpm(\n",
    "    bin_size=3, plotting_bin_size=10, num_minutes=10,\n",
    "    first_N_experiments=20\n",
    "    )"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Analysing the first ten minutes of the first ten sessions of each mouse"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "code_folding": [
     0
    ]
   },
   "outputs": [],
   "source": [
    "def plot_itpt_hpm(bin_size=1, plotting_bin_size=10, num_minutes=200, first_N_experiments=5):\n",
    "    \"\"\"\n",
    "    Aggregates hits per minute across all IT and PT animals. Performs regression\n",
    "    on the resulting data, and returns the p-value of how different linear\n",
    "    regression between the two animals are.\n",
    "    \"\"\"\n",
    "\n",
    "    # Getting all hits per minute arrays\n",
    "    ITshallow_train = []\n",
    "    ITshallow_target = []\n",
    "    ITdeep_train = []\n",
    "    ITdeep_target = []\n",
    "    PT_train = []\n",
    "    PT_target = []\n",
    "    num_itshallow = 0\n",
    "    num_itdeep = 0\n",
    "    num_pt = 0\n",
    "    \n",
    "    for animaldir in os.listdir(datadir):\n",
    "        animal_path = datadir + animaldir + '/'\n",
    "        if not os.path.isdir(animal_path):\n",
    "            continue\n",
    "        animal_path_files = os.listdir(animal_path)\n",
    "        animal_path_files.sort()\n",
    "        animal_path_files = animal_path_files[:first_N_experiments]\n",
    "        for file_name in animal_path_files:\n",
    "            result = re.search(pattern, file_name)\n",
    "            if not result:\n",
    "                continue\n",
    "            experiment_type = result.group(1)\n",
    "            experiment_animal = result.group(2)\n",
    "            experiment_date = result.group(3)\n",
    "            f = h5py.File(animal_path + file_name, 'r')\n",
    "            com_cm = np.array(f['com_cm'])\n",
    "            try:\n",
    "                e2_indices = e2_dict[animaldir][file_name]\n",
    "            except:\n",
    "                continue\n",
    "            ens_neur = np.array(f['ens_neur'])\n",
    "            e2_neur = ens_neur[e2_indices]\n",
    "            e2_depths = np.mean(com_cm[e2_neur,2])\n",
    "            xs, hpm, _, _ =\\\n",
    "                learning_params(\n",
    "                    experiment_type + experiment_animal,\n",
    "                    experiment_date,\n",
    "                    bin_size=bin_size\n",
    "                    )\n",
    "            xs = xs*bin_size\n",
    "            if experiment_type == 'IT':\n",
    "                shallow_thresh = 250\n",
    "                deep_thresh = 350\n",
    "                for idx, x_val in enumerate(xs):\n",
    "                    if x_val <= num_minutes:\n",
    "                        if e2_depths < shallow_thresh:\n",
    "                            ITshallow_train.append(x_val)\n",
    "                            ITshallow_target.append(hpm[idx])\n",
    "                        elif e2_depths > deep_thresh:\n",
    "                            ITdeep_train.append(x_val)\n",
    "                            ITdeep_target.append(hpm[idx])\n",
    "                if e2_depths < shallow_thresh:\n",
    "                    num_itshallow += 1\n",
    "                elif e2_depths > deep_thresh:\n",
    "                    num_itdeep += 1\n",
    "            else:\n",
    "                for idx, x_val in enumerate(xs):\n",
    "                    if x_val <= num_minutes:\n",
    "                        PT_train.append(x_val)\n",
    "                        PT_target.append(hpm[idx])\n",
    "                num_pt += 1\n",
    "\n",
    "    # Collect data\n",
    "    ITshallow_train = np.array(ITshallow_train).squeeze()\n",
    "    ITshallow_target = np.array(ITshallow_target)\n",
    "    ITdeep_train = np.array(ITdeep_train).squeeze()\n",
    "    ITdeep_target = np.array(ITdeep_target)\n",
    "    PT_train = np.array(PT_train).squeeze()\n",
    "    PT_target = np.array(PT_target)\n",
    "    fig = plt.figure()\n",
    "    ax = fig.add_subplot(111)\n",
    "\n",
    "    # p-val for linear regression slope similarity\n",
    "    p_val = linreg_pval(\n",
    "        ITshallow_train, ITshallow_target,\n",
    "        ITdeep_train, ITdeep_target\n",
    "    )\n",
    "    print(\"Comparing linear regression slopes of IT and PT:\")\n",
    "    print(\"p-val = \" + str(p_val))\n",
    "\n",
    "    # Some options:\n",
    "    # Order 1, Order 2, Logx True\n",
    "    sns.regplot(\n",
    "        ITshallow_train, ITshallow_target,\n",
    "        x_bins=plotting_bin_size,\n",
    "        color='forestgreen', label='IT shallow (%d Experiments)'%num_itshallow\n",
    "        )\n",
    "    sns.regplot(\n",
    "        ITdeep_train, ITdeep_target,\n",
    "        x_bins=plotting_bin_size,\n",
    "        color='cornflowerblue', label='IT deep (%d Experiments)'%num_itdeep\n",
    "        )\n",
    "    sns.regplot(\n",
    "        PT_train, PT_target,\n",
    "        x_bins=plotting_bin_size,\n",
    "        color='coral', label='PT (%d Experiments)'%num_pt\n",
    "        )\n",
    "    ax.set_ylabel('Number of Hits')\n",
    "    ax.set_xlabel('Minutes into the Experiment')\n",
    "    plt.title('Hits/%d-min of All Experiments'%bin_size)\n",
    "    plt.legend()\n",
    "    plt.show(block=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "code_folding": [
     0
    ],
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "plot_itpt_hpm(\n",
    "    bin_size=1, plotting_bin_size=10, num_minutes=10,\n",
    "    first_N_experiments=10\n",
    "    )"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Analysing the first ten minutes of the last ten sessions of each mouse"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "code_folding": [
     0
    ]
   },
   "outputs": [],
   "source": [
    "def plot_itpt_hpm(bin_size=1, plotting_bin_size=10, num_minutes=200, last_N_experiments=5):\n",
    "    \"\"\"\n",
    "    Aggregates hits per minute across all IT and PT animals. Performs regression\n",
    "    on the resulting data, and returns the p-value of how different linear\n",
    "    regression between the two animals are.\n",
    "    \"\"\"\n",
    "    \n",
    "    # Getting all hits per minute arrays\n",
    "    ITshallow_train = []\n",
    "    ITshallow_target = []\n",
    "    ITdeep_train = []\n",
    "    ITdeep_target = []\n",
    "    PT_train = []\n",
    "    PT_target = []\n",
    "    num_itshallow = 0\n",
    "    num_itdeep = 0\n",
    "    num_pt = 0\n",
    "    \n",
    "    for animaldir in os.listdir(datadir):\n",
    "        animal_path = datadir + animaldir + '/'\n",
    "        if not os.path.isdir(animal_path):\n",
    "            continue\n",
    "        animal_path_files = os.listdir(animal_path)\n",
    "        animal_path_files.sort()\n",
    "        animal_path_files = animal_path_files[-last_N_experiments:]\n",
    "        for file_name in animal_path_files:\n",
    "            result = re.search(pattern, file_name)\n",
    "            if not result:\n",
    "                continue\n",
    "            experiment_type = result.group(1)\n",
    "            experiment_animal = result.group(2)\n",
    "            experiment_date = result.group(3)\n",
    "            f = h5py.File(animal_path + file_name, 'r')\n",
    "            com_cm = np.array(f['com_cm'])\n",
    "            try:\n",
    "                e2_indices = e2_dict[animaldir][file_name]\n",
    "            except:\n",
    "                continue\n",
    "            ens_neur = np.array(f['ens_neur'])\n",
    "            e2_neur = ens_neur[e2_indices]\n",
    "            e2_depths = np.mean(com_cm[e2_neur,2])\n",
    "            xs, hpm, _, _ =\\\n",
    "                learning_params(\n",
    "                    experiment_type + experiment_animal,\n",
    "                    experiment_date,\n",
    "                    bin_size=bin_size\n",
    "                    )\n",
    "            xs = xs*bin_size\n",
    "            if experiment_type == 'IT':\n",
    "                shallow_thresh = 250\n",
    "                deep_thresh = 350\n",
    "                for idx, x_val in enumerate(xs):\n",
    "                    if x_val <= num_minutes:\n",
    "                        if e2_depths < shallow_thresh:\n",
    "                            ITshallow_train.append(x_val)\n",
    "                            ITshallow_target.append(hpm[idx])\n",
    "                        elif e2_depths > deep_thresh:\n",
    "                            ITdeep_train.append(x_val)\n",
    "                            ITdeep_target.append(hpm[idx])\n",
    "                if e2_depths < shallow_thresh:\n",
    "                    num_itshallow += 1\n",
    "                elif e2_depths > deep_thresh:\n",
    "                    num_itdeep += 1\n",
    "            else:\n",
    "                for idx, x_val in enumerate(xs):\n",
    "                    if x_val <= num_minutes:\n",
    "                        PT_train.append(x_val)\n",
    "                        PT_target.append(hpm[idx])\n",
    "                num_pt += 1\n",
    "\n",
    "    # Collect data\n",
    "    ITshallow_train = np.array(ITshallow_train).squeeze()\n",
    "    ITshallow_target = np.array(ITshallow_target)\n",
    "    ITdeep_train = np.array(ITdeep_train).squeeze()\n",
    "    ITdeep_target = np.array(ITdeep_target)\n",
    "    PT_train = np.array(PT_train).squeeze()\n",
    "    PT_target = np.array(PT_target)\n",
    "    fig = plt.figure()\n",
    "    ax = fig.add_subplot(111)\n",
    "\n",
    "    # p-val for linear regression slope similarity\n",
    "    p_val = linreg_pval(\n",
    "        ITshallow_train, ITshallow_target,\n",
    "        ITdeep_train, ITdeep_target\n",
    "    )\n",
    "    print(\"Comparing linear regression slopes of IT and PT:\")\n",
    "    print(\"p-val = \" + str(p_val))\n",
    "\n",
    "    # Some options:\n",
    "    # Order 1, Order 2, Logx True\n",
    "    sns.regplot(\n",
    "        ITshallow_train, ITshallow_target,\n",
    "        x_bins=plotting_bin_size,\n",
    "        color='forestgreen', label='IT shallow (%d Experiments)'%num_itshallow\n",
    "        )\n",
    "    sns.regplot(\n",
    "        ITdeep_train, ITdeep_target,\n",
    "        x_bins=plotting_bin_size,\n",
    "        color='cornflowerblue', label='IT deep (%d Experiments)'%num_itdeep\n",
    "        )\n",
    "    sns.regplot(\n",
    "        PT_train, PT_target,\n",
    "        x_bins=plotting_bin_size,\n",
    "        color='coral', label='PT (%d Experiments)'%num_pt\n",
    "        )\n",
    "    ax.set_ylabel('Number of Hits')\n",
    "    ax.set_xlabel('Minutes into the Experiment')\n",
    "    plt.title('Hits/%d-min of All Experiments'%bin_size)\n",
    "    plt.legend()\n",
    "    plt.show(block=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "code_folding": [
     0
    ],
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "plot_itpt_hpm(\n",
    "    bin_size=1, plotting_bin_size=10, num_minutes=10,\n",
    "    last_N_experiments=10\n",
    "    )"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Across-Session HPM Learning Plots\n",
    "## Analysing max HPM across sessions for IT vs PT mice"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 114,
   "metadata": {
    "code_folding": [
     0
    ]
   },
   "outputs": [],
   "source": [
    "def plot_itpt_hpm():\n",
    "    \"\"\"\n",
    "    Aggregates hits per minute across all IT and PT animals. \n",
    "    Looks at max hpm in 10 minute windows.\n",
    "    \"\"\"\n",
    "\n",
    "    # Getting all hits per minute arrays\n",
    "    IT_train = []\n",
    "    IT_target = []\n",
    "    PT_train = []\n",
    "    PT_target = []\n",
    "    num_it = 0\n",
    "    num_pt = 0\n",
    "    bin_size = 10\n",
    "    \n",
    "    for animaldir in os.listdir(datadir):\n",
    "        animal_path = datadir + animaldir + '/'\n",
    "        if not os.path.isdir(animal_path):\n",
    "            continue\n",
    "        if animaldir.startswith(\"IT\"):\n",
    "            num_it += 1\n",
    "        else:\n",
    "            num_pt += 1\n",
    "        animal_path_files = os.listdir(animal_path)\n",
    "        animal_path_files.sort()\n",
    "        session_idx = 0\n",
    "        \n",
    "        for file_name in animal_path_files:\n",
    "            result = re.search(pattern, file_name)\n",
    "            if not result:\n",
    "                continue\n",
    "            experiment_type = result.group(1)\n",
    "            experiment_animal = result.group(2)\n",
    "            experiment_date = result.group(3)\n",
    "            f = h5py.File(animal_path + file_name, 'r')\n",
    "            com_cm = np.array(f['com_cm'])\n",
    "            try:\n",
    "                xs, hpm, _, _ =\\\n",
    "                    learning_params(\n",
    "                        experiment_type + experiment_animal,\n",
    "                        experiment_date,\n",
    "                        bin_size=1\n",
    "                        )\n",
    "            except:\n",
    "                continue            \n",
    "            # Get running mean over 10-minute windows\n",
    "            hpm_5min = np.convolve(hpm, np.ones((5,))/5, mode='valid')\n",
    "            max_hpm = np.max(hpm_5min)\n",
    "            if experiment_type == 'IT':\n",
    "                IT_train.append(session_idx)\n",
    "                IT_target.append(max_hpm)\n",
    "            else:\n",
    "                PT_train.append(session_idx)\n",
    "                PT_target.append(max_hpm)\n",
    "            session_idx += 1\n",
    "\n",
    "    # Collect data\n",
    "    IT_train = np.array(IT_train).squeeze()\n",
    "    IT_target = np.array(IT_target)\n",
    "    PT_train = np.array(PT_train).squeeze()\n",
    "    PT_target = np.array(PT_target)\n",
    "    fig = plt.figure()\n",
    "    ax = fig.add_subplot(111)\n",
    "\n",
    "    # Some options:\n",
    "    # Order 1, Order 2, Logx True\n",
    "    sns.pointplot(\n",
    "        IT_train, IT_target,\n",
    "        color='lightseagreen', label='IT (%d Animals)'%num_it\n",
    "        )\n",
    "    sns.pointplot(\n",
    "        PT_train, PT_target,\n",
    "        color='coral', label='PT (%d Animals)'%num_pt\n",
    "        )\n",
    "    ax.set_ylabel('Number of Hits')\n",
    "    ax.set_xlabel('Day')\n",
    "    plt.title('Max Average HPM')\n",
    "    plt.legend()\n",
    "    plt.xticks(np.arange(0,18,2))\n",
    "    plt.show(block=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 115,
   "metadata": {
    "code_folding": []
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "No handles with labels found to put in legend.\n"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAYUAAAEWCAYAAACJ0YulAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4zLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvnQurowAAIABJREFUeJzsnXWYHdXdgN8r65bdjW/cTtw9hCg0uHuLFIe2FNqPooUCbUoFihT3FkJxglsSIK7Ec+KyyWaT7G7W7cr3x5krK1d299puzvs895mZO3NnzrX5nZ+bnE4nGo1Go9EAmKM9AI1Go9HEDlooaDQajcaNFgoajUajcaOFgkaj0WjcaKGg0Wg0GjdaKGg0Go3GjRYKGo1Go3FjjfYANCcmQoi9QFegq5TymNfzPwEjgN5Syr1huG4KkA/8IKU8PdTnjwZCCCfQX0q50+u5B4F+Usqfex1TATiBYuB/wP9JKe3R+i40sYnWFDTRZA9wmWtDCDEMSArzNS8EqoFThRBdwnEBIUSsTrZGSClTgVnA5cD1Xvui8V1oYpBY/fFqTgz+A1wJPGVsXwW8ATziOkAIcYax3Rc1w31ZSvmgse8SYC4wUkpZIoQ4DXgVGCalPOrjmlcBzwGnAVcA/zDOdRcwVkp5ode1nwBMUsrfCCEygMeA0wGHcZ0HjJn21agb7Erj/M8IIV4FXkTNtJ3AV8CtUsrjxrlHAy8D/YAvjXPukFLeZ+w/03jfvYAtwE1Syg3BfrD+kFJuE0L8CAz1ejrgd6E5MdCagiaaLAfShRCDhBAW4BLgv/WOKUfdrNoBZwA3CyHOBZBS/g9YBjwphMhG3WSv8yUQhBA9gOnAm8bjSq/d84DThRDpxrEW4GLgLWP/64ANdRMfBZwKXOf1+gnAbqAj8GfAhBJYXYFBQHfgQePc8cCHwGtAlnHt87zGORp4BbgRyAaeB+YLIRIae19NRQgxGJgKrPN6OpjvQnMCoDUFTbRxzVC/B7YBB713SikXeW1uEELMA6YBHxnP3QpsABYBn0gpP/VzrSuBDVLKLUKI48DfhBCjpJTrpJT7hBBrgXNRM+SZQIWUcrkQohNKs2gnpawEyoUQjwM3oG7YAIeklK5Ztg3YaTwAjgohHgMeMLYnov57T0opncAHQoiVXuO8HnheSrnC2H5dCHGP8brvfby3tUIIh9d2IvBeI8fYgULgJZS2443f70JzYqCFgiba/Af4AeiNuhnXQQgxAfgrytQRDyQA77r2SymPCyHeBe4ALghwrStRJh2klIeEEN+jzCSuGfNbKLv6Gyibu0tL6AnEAXlCCNe5zMABr3N7ryOE6Ag8iZqRpxnHFxm7uwIHDYHQ2Ot7AlcJIX7t9Vy88TpfjG7M0ezvmEbw+11oTgy0+UgTVaSU+1BOztOBDxo55C1gPtBdSpmB8geYXDuFECOBX6JMME/6uo4QYjLQH7hbCHFYCHEYZfK5zMsx/C4wXQjRDWXOcQmFAyjndHspZTvjkS6lHOJ1ifrlhucazw2XUqYDP/cadx6QI4QweR3f3Wv9APBnr2u1k1ImSynn+Xp/oSCI70JzAqCFgiYWuBaYKaUsb2RfGlAopawSQoxHzeABEEIkouze9wDXoG60t/i4xlXAN8BgYKTxGAoko0xDGL6IRSizyh4p5Vbj+Tzga+CfQoh0IYRZCNFXCDHNz3tKA8qA40KIHOD/vPYtA+zAr4QQViHEOcB4r/0vAjcJISYIIUxCiBQhxBlCiDQ/1wsV/r4LzQmAFgqaqCOl3CWlXO1j9y3AQ0KIUuCPwDte++YCuVLKZ6WU1ajZ+CNCiP7eJzCEx8XAU1LKw16PPSiTyVVeh78FzMajJbi4EmXC2YIyA70H+Atp/RMwGhUx9RleM28pZQ1wPuoGfNwY96cobQTjs7geeNq41k7gaj/XChkBvgvNCYBJN9nRaKKPEGIF8JyUsr7zV6OJKNrRrNFEAcP0JIFjqHyJ4ah8BY0mqmihoNFEB4EyhaUCu4ALDd+FRhNVtPlIo9FoNG60o1mj0Wg0blqd+WjChAnOnJycaA9Do9FoWhWbN28+JqXsEOi4VicUcnJy+OADnVej0Wg0TUEIsS+Y47T5SKPRaDRutFDQaDQajRstFDQajUbjptX5FDQajeZEp7a2ltzcXKqqqhrsS0xMpFu3bsTFxTXr3FooaDQaTSsjNzeXtLQ0evXqhcnkKbbrdDopKCggNzeX3r17N+vc2nyk0Wg0rYyqqiqys7PrCAQAk8lEdnZ2oxpEsGihoNFoNK2Q+gIh0PPBos1HGo1G0xpxOKCiBCrLwOmE+ARIyYC4lrXyDrtQMJqAr0a1Hzyz3r4EVNu/MUABcImUcm+4x6TRaDStGocDjuWCrcbznK0aKkohs1OLTh0J89FtwFYf+64FiqSU/YDHgUcjMB6NRqNp3VSW4qytbmSHE2fhYXA6mn3qsAoFo9ftGcBLPg45B3jdWH8PmFWvb61Go9FovCkvIbHgAAXlVdSvcu10OikorySxqrTZpw+3+ehfwJ2ofrWNkYNqUo6U0iaEKAayUY1HNBqNRlOf4qN0W/MRuc5zOJrR0FSUWJxPN2cFMLFZpw+bUBBCnAkckVKuEUJM93FYY1qBbvCg0Wg0vkjNJK6mgt7L5/k+ZuYVzT59OM1HU4CzhRB7gbeBmUKI/9Y7JhfoDiCEsAIZQGEYx6TRaDStm/Qs6DvS936TGYZPa/bpwyYUpJR3Sym7SSl7AZcCC6SUP6932HzgKmP9QuMYrSloNBqNP067DpLTG993ypXQLmDbBJ9EPHlNCPGQEOJsY/NlIFsIsRO4A7gr0uPRaDSaVkf7HLjh75CQ5Hmu93C4/F6YfE6LTh2R5DUp5SJgkbH+R6/nq4CLIjEGjUajaVMkp0N1pVrvPwauuC8kp9VlLjQajaY1cmS/Z71jj5CdVgsFjUajaY1ooaDRaDQaN3WEQs+QnVYLBY1Go2mNuISCyawczyFCCwWNRqNpjbiEQnYXiIsP2Wm1UNBoNJrWRnkJlBWp9RD6E0ALBY1Go2l9HA2PPwG0UNBoNL7YvhpevU8tNbFFfngij0B3XtNoNL5YOA/ydkNNJQwYG+3RaLw5ss+zrs1HGo0mIriyZV1LTezgcjJb4yGrc0hPrYWCRqPRtCacTo9Q6NANzJaQnl4LBY1Go2lNlBRAdYVaD7HpCLRQ0Gg0mtZFGP0JoIWCRqPRtC7CVN7ChRYKGo1G05oIYzgqaKGg0Wg0rQuXppCQDOnZIT+9FgoajUbTWnDY4egBtd6pJ5hMIb+EFgoajUbTWig8DPZatR4G0xFooaDRaDSthzA11vFGCwWNRqNpLWihoNFoNBo3Yc5RgDAWxBNCJAI/AAnGdd6TUj5Q75irgb8DB42nnpZSvhSuMWk0Gk2rxhWOmpoJyelhuUQ4q6RWAzOllGVCiDhgsRDiCynl8nrH/U9K+aswjkOj0WhaP7U1UJin1sOkJUAYhYKU0gmUGZtxxsMZrutpNBpNm+bYQXA61Hqn0GcyuwhrPwUhhAVYA/QD/i2lXNHIYRcIIU4GtgO3SykPhHNMGo1G0yqJgD8BwuxollLapZQjgW7AeCHE0HqHfAL0klIOB74FXg/neDQajabVEoHII4hQ9JGU8jiwCJhT7/kCKWW1sfkiMCYS49FoNJpWh1somKBD97BdJmxCQQjRQQjRzlhPAmYD2+od08Vr82xga7jGo9FoNK2afMN8lNkJ4hPDdplw+hS6AK8bfgUz8I6U8lMhxEPAainlfOA3QoizARtQCFwdxvFoNBpN66SqHEqOqfUwmo4gvNFHG4BRjTz/R6/1u4G7wzUGjUajaRMc8Yq/CbNQ0BnNGo1GEy22r4ZX71NLf3g7mcMYjgphDknVaDQajR8WzoO83VBTCQPG+j4uQuGooDUFjUajiR7VlXWXvnBpCmYrZHcN65C0UNBoNJpYxun0RB61zwFLeA08WihoNBpNLFN2HCpL1XqYTUeghYJGo9HENhHKZHahhYJGo9HEMhGMPAItFDQajSa20ZqCRqPRaNy4wlHjEiGjQ9gvp4WCRqPRxCoOhyebuWN3MIf/lq2Fgkaj0cQqxUehtkqtdwy/PwG0UNBoNJrYJT9ymcwutFDQaDSaWCXCTmbQQkGj0WhilwiHo4IWChqNRhO7uCKPktMhJSMil9RCQaPRaGIRWy0cO6jWO/YAkykil9VCQaPRaGKRwjxw2NV6hCKPQAsFjUajiU2i4GQGLRQ0Go0mNolCOCpooaDRaDSxSR1NoXvELquFgkaj0cQiLqGQ0QESUyJ22bC18BFCJAI/AAnGdd6TUj5Q75gE4A1gDFAAXCKl3BuuMWk0Gk2roKYKivLVegRNRxBeTaEamCmlHAGMBOYIISbWO+ZaoEhK2Q94HHi0xVfdvhpevU8tNRqNpjVy9ADgVOttRShIKZ1SyjJjM854OOsddg7wurH+HjBLCNGyYNyF82DfZrXUtG30BEDTVolS5BGE2acghLAIIX4CjgDfSClX1DskBzgAIKW0AcVAdosuWl1Zd6lpu+gJgKatUkcoRC5HAYLwKQghbgNeBUqBl4BRwF1Syq8DvVZKaQdGCiHaAR8KIYZKKTd5HdKYVlBfm9BoGkdPADRtFVc4qskM7XMieulgNIVfSilLgFOBDsA1wF+bchEp5XFgETCn3q5coDuAEMIKZACFTTm3RqPRtDlcmkJ2F4iLj+ilgxEKrtn86cCrUsr1ND7Dr4MQooOhISCESAJmA9vqHTYfuMpYvxBYIKXUmoJGozlxqSiBsiK1HmF/AgQXkrpGCPE10Bu4WwiRBjiCeF0X4HUhhAUlfN6RUn4qhHgIWC2lnA+8DPxHCLETpSFc2qx3odFoNG2FKPoTIDihcC0qpHS3lLJCCJGNMiH5RUq5AeV/qP/8H73Wq4CLgh+uRqPRtHGiGHkEwZmPvpFSrjX8AkgpC1A5BRqNRqMJNVEWCj41BSMjORloL4TIxONHSAe6RmBsGo1Gc+LhijyyxEFW54hf3p/56EbgtygBsNbr+RLg3+EclEaj0ZyQOJ0eTaFDdzBbIj4En0JBSvkE8IQQ4tdSyqciOCaNRqNpyPbVsOQjmHIuDBgb7dGEh5ICqK5Q650ibzoC/+ajmVLKBcBBIcT59fdLKT8I68g0Go3Gm4XzIG831FS2XaEQZX8C+DcfTQMWAGc1ss8JaKGg0Wgix4mQwX4kOo11vPFnPnrAWAYMP9VoNBpNCIhyjgL4Nx/d4e+FUsrHQj8cjUajOYFxCYWEZEhvWW3Q5uIvTyHN6/H7ettp4R+aRqPRnEA47HA0V6137AGmlnURaC7+zEd/cq0LIc713tZoNBpNiCk8DLYatR4lfwIE309BF6nTaDSacOLtT+gUHX8ChLnJjkaj0WiCJAbCUcG/o3kjHg2hnxBig7FuApxSyuHhHpxGo9GcMMRAOCr4z1M4M2Kj0Gg0mhMdl6aQmgnJ6VEbhj9H8z5f+zQajUYTYgry1DKKWgJon4JGo2mMmmpVTgKgpgpstdEdT1vHbgOn0btMCwWNpo2wfTW8ep9atma2LofHroOy42q7rAgevx52rY/uuNoy3kI3VoWCEOI7Y/lo5Iaj0bRiFs6DfZvVsrVycCe8+w+oKqv7fHkxzPsLHDsYnXG1dexeQiGK4ajg39HcRQgxDThbCPE2niY7AEgp1zb+Mo3mBKUtFGxb+rHKrG0MWw2s+BTOuDGyYzoRsNs86x26R28c+BcKfwTuAroB9escOYGZ4RqURqOJEns3Bdi/OTLjONFwCYXMThCfGNWh+Is+eg94Twhxv5Ty4QiOSaPRRItAnb6i0AnshMClnUWpMqo3/jQFAKSUDwshzgZONp5aJKX8NNDrhBDdgTeAzoADeMHo5uZ9zHTgY2CP8dQHUsqHgh++RqMJKf1Hw9pv/OwfE7mxxDAL8vN4YZfkhr6CmZ26hO7EUXYyQxDRR0KIucBtwBbjcZvxXCBswO+klIOAicCtQojBjRz3o5RypPFoWwKhrUSjaIKiwm6vs2yVTDkX4nyYLxKSYfzpkR1PjPK43MSKgqM8LgOY25pKaxAKwBnAKVLKV6SUrwBzjOf8IqXMczmjpZSlwFYgpyWDbXW0hWgUTdAU1VTXWbZKsrvCzMsa39e5F6RnRXQ4sUqZzVZnGTKiHHkEwecptPNaz2jqRYQQvYBRwIpGdk8SQqwXQnwhhBjS1HPHNG0hGiVWqSj1NDivraobvRElHPWWrZbcHXW3XX6EfVvg0K7Ij+dEwWyFrBCaopo7jCCOmQusE0K8JoR4HVgD/CXYCwghUoH3gd9KKUvq7V4L9JRSjgCeAj4K9ryaE5hVX6rkqvJitV1aBE/eAgd3+H+dJjCV5bDNmLu5hIF3HZ4f3o38mE4U2ncFa1y0RxFYKEgp56F8Ah8Yj0lSyreDObkQIg4lEN6UUn7QyLlLpJRlxvrnQJwQon0Txh8+YskfEEtjiTZyFXz2vKcZiYvio/DfhzxZuJrmsWWJJ5EqIVkt4xM9sfPbVkC+LosWMpxeemUM+BMgSPOR4R+YL6X8WEp5OJjXCCFMwMvAVl/9nIUQnY3jEEKMN8ZTENzQw0ws+QNiaSzRZnGDuYWHyjJY4ydyRhOY9Ys86wlJnvWpF3rWf3wvYsNp83j7JGIgHBWCCEltAVOAXwAbhRA/Gc/dA/QAkFI+B1wI3CyEsAGVwKVSytjo8hZL/oBYGks0sdvhwDb/x+zbDFwUkeG0OQrzYP9Wtd5rKJR4zc+GToFFb6tjNi2B6ZdC+8jGjVTY7SR7LdsE9tipeeQibEJBSrmYeqUxGjnmaeDpcI1B08YwASZzXZW7PpZwznPaOBt+8KyPmA4/vu/ZNltg6gXw8dOAU2kL590W1GlDFdNfVFNNstcymkws2M/cfWv5uOfolp3IO0AiRoSCX/OREMIshAhxIG4YObzHU8jLrkv9tjnMFug3yv8xOrmqeTidsH6hWrfGw6BJDY8ZPg3adVTrG35QjeaDIFQx/bEU3XXNvtVMLDvCNfta6OdzV0c1eT7bKONXKEgpHcB6IURsiDBfVFfCvLnw3B0qVBGg+Bi8+Yg2ubQ1pl3su9RCSjs1w9U0PTjhwDYoylfrgyZCYiNzcYsVTjpfrTsd/v07XoQtpj+KJBmTzqSWTD4dDo+mYLGCOTY6GQQzii7AZiHEd0KI+a5HuAfWJD55BuTKhs/vWAMfPRX58WjCR7cBkNO/8X1mc8z8saJOU4MTvB3M/gTryJmQlq3Wf1oIx482d4RRYUF+HpcuXcSC/LzoDcJWq6rRPnkLqrYoallVEb0xeRHMP+hPqH7NDwH/9HrEBkX5yvHli63LdA34tkRRPuRuV+sujcFVlqG0EJYHLMvVgJi4UYSapgQn1FbDpsVqPTUTeg/3faw1TpXCAHDYYMmHAU8/sWA/8+S3TCzYH3gsYSZs5SmCxW6Hd/4GX78Gx/O9nrfB6/fHhGUjmDyF74G9QJyxvgqVdBYb7N+KR9r6O0bTJlj5ucfRnJSqlikZyg4OygHaxFyFqN8ooo1c7ckOH34yWAJUQh1zijLVAaz9VgljP4TM/h4Com7K2rzYt0kvbzcsi74RJpiCeNcD7wHPG0/lEEuZx8FEm+iIlJijWbPz6kp1EwJISvPE0ZvNntlrTVWT8zmG5u9mnvyWofm7m/S6NoPLwQwwYkbg4+MSYPI5at1umEL8EBL7e1vhp4X+93ub8aJEMOajW1E5ByUAUsodQGy4yUGpuhY/qeEmM/QdGbnxaIKiWbPznxZ6ZrRjTqVOxPPkc5XpA5TgaELWbSzNZCNO2XHYuU6td+4TfEG2sT9Tghlg9VeekiOhxulU3+dzd3icsg57QO0kZgn0OZVHPyM/GKFQLaV01xQQQlgJaK+JICnpMOls3/uj3MVI0zhNVuMdDtUKEpQvYdycuvsTkmDm5Wrd6YBvXg96LCf0THbjjx5zXFMitxKSPP+72urwmT0+fwHm/1uFm7twOuHFP6gIw9ZGoIJ3raQg3vdCiHuAJCHEKcC7wCfhHVYTmXm5yrCMT2q4r7oC/vco1NY03KdpPexcq7JpAQZPhoxGSmSNnAGdehnHr/PMgDW+cZkrTGYYNrVprx1/GiSmqPWVn3vCwUPFgW2q+GFjlByD794M7fUiwdiftWx/BAhGKNwFHAU2AjcCnwP3hXNQTcZshumXwO9ehjSj3nu7jtC+m1o/sE2FrTpjR8HRNBHvqKKJZzZ+jNkCp17t2f76Nd9N6DXKxHbY8KP0GwWp7fwfX5/EFJhgtFapqYIVn4V2fPXs651r6oVsbl7S+iZ7vYeqwIjGGHoSjJ4d2fE0QjDRRw7gdeBhVHjq6zFTn6g+CUnKCQbqBnH5PR6754bv66bta1oPR/bD7vVqvdsA9fBF3xGerOYj+2Hdd+EfXyiJZEXcOrkJQTiYG2PCmR4T7YpPQxtrXy+KLM6wWse5zF32Wo+PqbWw5huPXyG1nccfmpoJ598eEz2wg4k+OgPYBTyJqlO0UwhxWrgHFhKyusCld6nmFQAL3oQty6I7Jk3T8dYSJvjQErw59SplDgFY8FZMxH4HTaQq4trtaqIEqkS2GNu88ySneVp0VpUrM1Ko8GFfT3IYfqjEVM+krzVQUaLuQaBu/lc97DGDxifGTOJlMKP4JzBDSjldSjkNmAE8Ht5hhZCeg+Gsmz3bH/xLd49qTZSXeG5eadkwuJGaPPXp0B3Gnmq8vjjocgwxQaQq4u7ZAGVFan3IFI+G3RwmnuXJE1k2PzRjr61Rmp4XLvNEksskOHp24JyKWGLBPFXeHZQJtEO36I7HB8EIhSNSyp1e27uBI2EaT3gYNROmnKfWbTUw7y91ywJrYpe133ga6ow/Lfick+mXeprELJvfOiNVwom36WhkM01HLlLbeRyklaWw5uuWna+yDP77JxVc4EWVSQmAeKcDuvZT33FrIW+P53NJzYSTL47uePzgUygIIc4XQpyPqnv0uRDiaiHEVajIo1URG2GomPVzGDhBrZcWKsFQUxXWS1bY7XWWbYGIloSw22DlF2rdGq8yaYMlJcPTGMZWA9/9N/Tja61UVcDW5Wo9sxN0H9jyc04512MfX/KRClNtDsXH4JV7VD9oUK1Az7sNxpxKpfeEYOAEiG+BdhNJnE744kVP6O/sXzRecDBG8KcpnGU8EoF8YBowHRWJlBn2kYUas1n9uDr3Vtt5u+HDJ1T8e5goqqmus2wLRLQkxJZlUGpodCOm1+0VHAwTzvAq9fy97uHsYusyj/Y1YgaY/LY9CY60LE/kTPlxT+Z5U8jfBy/dBUcPqO3MTnDtX9V3f9bNVLpMVODpI90a2Pijp9ROtwGqBHkM41MXl1JeE8mBRISEJLjsHnjxTmVP3bocFr6ltIgwEEv130NFRGvH1HEwn9H018fFq1nZe0b9xq9eg2seCc1NsDXjbToK5Q1qynkqusZhg8UfqqzzYBvR79kEb8/1RBN17QeX31snTNZh8prDHtoJx4/ETA8Cn1RXeiVSmuC062PGoeyLYKKPegshHhNCfBCzpbObQkZ7uOxurwJq78dEvRFNPQ5IOGhUQ+0zovldqYZM8YSw7t/iMZvEKGE3OR4/AnsNLa/HIMjqHLpzt+sAI6er9dIC+GlBcK/btFj5EFwCod9ouOqhwHkTrSGS8Id3PSU5Rs+CnH4hv0SoTbrBiKyPUFVSnyIWS2c3h5z+cO5vPNvz/60rqcYaK4JIVgsGkwl+5qX0fvOGV7er2CPsJkdXJBeEpyHRSRd4woEXf1C33WRjLJuvNDnXcSNnqklbQiPVCQzcSVKxLhQKDsEyo/hDYkrYLBKhNukGIxSqpJRPSikXSim/dz1CcvVoMnQKzLhMrdtt8PZfoah1BVU1hVbVM6D4mOcPn9VFzRxbQveBSmMAKDoMq75o2fnCSFhNjk6np0qnJQ4GTwn9NbI6q/LboLQS777P3jgc8NWr6uHi5IvgnF8FjDCrdiV45crYjiL88hVlSgN1r/GVydxCQm3SDUYoPCGEeEAIMUkIMdr1CMnVo83JF8Ew4wdcUQJvPRIz3Y9CTavqGbDqS095iglnhsYGO/sXnpvN9++Gvk5PayB3u6d+1MDxkJQSnutMvRB3Bdsf38NUv36mrRbef9xTRM9khjNvVjXMgvD3VHpXRY5Vc+D21arzIyjT59g5/o+PIYIJ+h4G/AKYiWcC4zS2fSKE6A68AXQ2XveClPKJeseYgCeA04EK4GopZeQa+JhMcPatavaYu11FPbz7dxg6VcVbgyeMrJUT9eYiwVJTDWu+UusJyS2PoXeR2UkJmKUfQVWZsvXO+WVozt1aCEVZi2Bon6M0s82LoTCPpDjDFOR0QmU5/O+vHr+GNR4u/J0SUkFSaYkjs7YacMKWpc0LQggntlqlJbg47bpWlWQXzBTsPKCPlHKalHKG8fArEAxswO+klIOAicCtQojB9Y45DehvPG4Anm3C2ENDXDxcejdkdFDbu36Cj5/yZB4W5cdEN6QTho3fez770af4tS03makXesJaV36ubL4nCrZaT8vNlIzw9xiZeoF7Nd2m8oEsdhs8ebNHICSlKYdyEwQCgN1khh5GbsW+rU3utBd2ls2vW9G397DojqeJBCMU1gNNLJ8IUso816xfSlkKbEV1bfPmHOANKaVTSrkcaCeEiHxB8dR2cNYtvvd/9WrT1NSifPjoKY/zzG5Tcdu6Sqt/nE5PGKrJ7KmpEyqSUlQ1XVDmqW/eaPk5bbXKofqvG72awDiUOTKW2L5aaUigTKbhnrnu8pQtjzN+9x1rKz0aeLtOcO1c6C6ad353uRNnbJmQio8pLRSUFuRdtbeVEIxQ6ARsE0J81dyQVCFEL2AUUD/jJAc44LWdS0PBERn2bfa/f9H/lB06ULJbwSGVB1E/HG/+v+uqlJqG7N7gSVwaOB4ywxCDPuZUyDZ+YttWeGatzcFhNGH/9j/KqerC6VBZubHkt4iU6QhUpYAf3mvwtNXbt3DebcrM1FwGedXAiqUopG/e8GRzT71Qhem2MoLxKTzQkgsIIVKB94HfSinrT58a8ypFZzp9YJsoBo7mAAAgAElEQVT//fl74W9XqhlsSobnkZxed3vNN+5Zorm+ZrDiUxUG2LVvWN6CPyYW7GfuvrV83DM2YgQaHc9yr95NwVRDbQ4Wq6qiOu8vavur12j2T27LMt8lro8dhMXvx8ZMsbzEy+nZEzr3Cu/1DsjAJa3zdkLPQc2/RkZ7yBmgcln2blLvMaWJGe+hZu9m2PSjWm/XydPHupURUCi0JPxUCBGHEghvSikbK1WZC3T32u4GRMfQG2zmpdOhsqFdFSb9kFOr/hhZNq8aSxt/iIpQuGbfagaUHSNr32pUr6To0mA8BYc8N67OfVR123AxYKyy8+7ZCHm7SLYYNXQcdlVzJ9C1HXYVvrzkozpP59SU1z1uww+xIRQ2LfZEc42YHv6MbkfdYAYnavZXbraS4toXiuS8wZOUUHA6QK6MboMaux2+eMmzPeca5a8MQIXdTrLXMhYIKBSEEKV4plLxQBxQLqX0K5aNyKKXga1Sysd8HDYf+JUQ4m1gAlAspYxOIL0Y7799Y1YXlRlbXmw8StTSETiaJ8W7+9e2lcrJ13tY8BU/Q0Cs9SFuMB7vrl0TzwjvjctkUlnSezYCkGE36gA5nfDa/XDxnTBogtouPgpHDqgyzkf3G8tcT+0gL1y22HTX+WLFr7DeyE0wmT0h2OEkp7/KgzC+2yPWRDrZqii0JpBcY1PmgVAI/cGTPCUktiyNrlBY87WyJoD6f4vgnOdFNdUkey1jgWA0hTpdLIQQ5wLBvOMpqFDWjUKIn4zn7gF6GOd9DtXa83RgJyokNXr1lkZMVzemY7kN98UlwMX/5ymm58LpVHkN5ceVgCgphI+edP8ZKswWkuu3gyw6DP99SEVeDJ6kQvd6DYmJjktRo7Ic1hk+mJQMFRIcTmqq68zyXbbuRIddzTrffww69VQ3/5rgewO4ZsQZLkGX3TWEg24mdpuqEwRKEKZnhf+ayenYRs/GuuoLnECN12/bBNT0HEJ8Tv+WXyezE3Tpo4pb7t6ootaSUlt+3qZSUaKaOYH6H8+5NuhJTSzWR2vyVFVK+ZEQ4q4gjltM4z4D72OcwK1NHUNYiE9U4XGfPgdyFW7lyGKFKx9sKBBAffFJKerhcprlbnPPegusiSTXlJNvzJTq4Ko7v+ZrSGmnBMTQk1T2rStZa8syWPKhJ7yttEj9Abr0CfnbD0SNw8Hnhw5wtEq9j5LaGo7X1NAuPrCKHJB130Kt8fmMnRO8Ka+57FzricTxooPrO7LV+K6ompCsmvh07KEmC17lOIos8WTZvTSIAWNCOerm4d3wJlQ5H0HwQNcRjMrexoUFe+o8vyStEy/0nMRrBLg5BMvgyeo/4bCp/20E36ObBW95fk8x3DwnWIIxH53vtWkGxhItZ3C4SctUdVeKj8HLd6kU+owOTas3P/MK9SP1qqVUY7aoGcT5v1UlhjctVuquq1dr+XFVemHVF6q72JDJKsppZb1G6LVV8PLd8IsHwmtzr0dhdTVXLv+BzSWeePDCmhpmLvyCNyaczNB2za+kbsLpaeFosXqatYSTYMw6Fit06uURAB27KydtenbdWWBaFnyrQlvLLXF1hcKGH5TDPD07tONvCq5m9/FJQZs0WsrR6irePXiAt3pN4pnOQ3hl5yIAbCYTPx8wC8orWFNUwNis9i2/2KCJnl4ZW5ZFXCjEO+ywunU0zwmWYDSFs7zWbajieK3TrR4sGe2pMFma5wBKSFIax+Yl8JnheDKZ4ZYnPNpEz8Fw2rUqWmHzEvVjdsVvlxbUjcKpj60GPn8RbnosYiWg7924xi0QZhQf5IbDW3mh8yAWZuRw46olLJp1OnHNLEWRaLd5wjmHTlWCOdxk1w2FLDHHke6opdgS5zH9XPNnT3VVf5x0nvIPrf4Ktqh4+UqzVfURLi2EeXPVuSLcEMb9u3WFUA+ZHLExbDpeRK0RebcnMR2nWyfw/F5/PJIfGqHQPkcJ6yP7VOJpVUWEGtio99euttK9HuvNc4IlGJ9C2+urEAQtcQDtq6riX7ZEfmP8CWzAW2VVXJbtxOS6kZst0Ge4epx+vXJ6blqsYueryn2fHJRD68h+ZfcOM4crK/k676B7+/ZDGxhWUUTKoVoWZuRwqKqSbw8f4rSuzVOZ02xe1UCbWg3Vle3c1KznnoOVBmDkRBRb40mvqaXEEq+EQtd+ylkaLDn91GObakhYEJ9Mt4ws9T3l7VLNnC76fUTr6Df43TajImpzI2Pig3ifT+3YwtqiAs7M6c7POue0zAw5eJISCvZaFcE2LIw+qe1rVKixET2V4PIZtoLmOcHiUygIIf7o53VOKeXDYRhPzNBcB9CuslIuWryAotoavIpzc+/GtcjSEv40bFTDF1ms0G+UethuUjeRzUv8Xyhvd0SEws6ykjqfQaqRtZvqVRJ5e2kxp9E8oeD+U/Uc3HRfyYzLYOnHTY8HN5vVTfr1B5Tpzpv0bLjg9hZpYU6TSTWIcTdzWgYL58GsK5p9zqCx22HHGpzeAQ4ZHaBH082NzZ0Yjc1qT2ZcPEW1DSO0XDiBxcfyWXwsn/s2rGFqh06c0bU7p3TOIT2urk+p2m7ni7xcRrnyfurn/wyeBIveVutbloZPKKz9ViWhAua4ep/IkCkx3zwnWPy9i/JGHgDXAn8I87haLXO3rPf5Z3hj7042Hg+Q32CNC872+9GT8J+HlHMtjC1FM4KItU4P4piG1PtjNydZbcBYuPphtWwqHXvArU/A7Cs9AsBkhpufCE3UUEZ7uPQur2ZO74W/mVPhYXj2Nnh7LsnePSMSkppV2LG5E6MEi4UrevkW8KMzsxnu5YeyOZ0sPHKY3/+0inFfz+eGlUuYf3A/5TYbPxUVcvJ3n3P7upU4DGFgczr5+9aNOF3CoUN3j2l2x9rw9F6vqsD55cvuzbR6od3O5Z94ckFaOf7acbob6Qgh0oDbUCGjb9Pam+yEieKamoD9Cj4+uI9hgRyzgyaqiKT6s9j67FqnHpmdYNxpMGpWyEPyhmS0o2tiEoeqGg/NNAGndg7yJupwKNv7is+weFdrTc9uclG0kJCcrnwCSz5W22ZzaMtJdxugmjm99w+1Pf/f6rvq0YJMXl/Y7ar0+zFl6kt2eN20juyH795UmdwRwOF0sqLgWKP7bu0/iNvFECwmE/vLy/jsUC6fHjrAFsNnVeNw8E3+Ib7JP0Si2YLd6XD7J7x5Zuc2cpKSubxXXyXUB09WNYdsNUowDJnc6PWbndkvV2LyEjZpxufrQM2sTcXHVHG+3kObdt4YxK++I4TIEkI8AmxACZDRUso/SCnbbjeaFlBiqw0YllVY41uldhMXr/IiGrOVt++mSvF29WrrV5QPX78Gj10HnzyrGqB7U17i0SaaWKytsKaaaj+aiBN4XG52z+L88tnz8PkLUHCQFO+bVmWZmuW2RYZOgemXqvVwNnPavtotEAASDM2g2tUFbfWXdcNTw8j7B/ayqlAJhaEZmVgMTcxqNvH7gUPd2z1SUrm5/0A+m3YK382Yw+1iCAPSPDmxVQ57owLBxYu7t3u0hSBrIV2zbzUTy45wzT4f5Ul8UFTsaebjwOMyL7Z4tOSK0sBVDloDPoWCEOLvwCqgFBgmpXxQStk23nWY6JiQSGIAu2Lf1DS/+930HAy3Pg3TLvGky6dkwA3/UPXjr/8bXPeoylA1GwpfbbXKe3j2tyozd8sylRT2+HUe84HTAY9d77sjlhfVdjs3r1pKgdEaMrOemSjBeK/v5+7jT5t+8vxBGyN3uxqbQardM9Oitjo0FUubSGltLa/t2YHdEHoOp5PycPSbmHaxJyEvXM2ccmWdTddNq8L126ip8hQbDCOF1dXM3bIBUDeXucPHeIIr/GQm9ElN4zcDBvPV9J/x1fRT+XX/QaQEyPjfW17Gewf2qtalnXtBptFvesdqT1G6ejQns7/KbudtRzxPdh7K850G1rlpllri3BPB9dYQlnmPIv7uYL8DugL3AYeEECXGo1QIESP5+43Q3IiUFuJ0Onl9706q/Myq400mLureK/iTpmfBjEtV7gKoxClXWKHJpMwTF9wOt7+gZqOpXmapvZtUBc+PnwJbbd3ifLYa+PBfvhO0jPdz/8a1rC5SM6T+qel8P+t0rGbXrM/MGxNPJtHIVn1j707+sc1PxdF6QsiV41ruumltXxXRrne7y0qZs+grJcyM5xxOJ6d//zW5FQGiv5qKyQTn3KoKuIG6Ob//WGht0D60ALdQgPAnBQJ/3brB7VO7unf/ZuWwDEjL4I6BQ5ndKXAV/TvXr2b0V/M5/YdvWNLRqClWU6XCU704VFnBnT+twmb8P20OBy/sku5tb8pstfxw5DD/2LaJi5YsZMSXH/G30ipe7ziAi4/tBjweMbvJzN9yRrAstSOlmZGv+h8O/PkUWqcrvbkRKS2g2m7n3g1reD93n9/jLuvZl46JYRBWaZmqT8DUC1Rt+RWfNaj62tUozmd1awxO1VT8wjsaPeXLu3fw7oG9ALSLi+fF8VNIi4vjuNdsb3x2B54bN5nrVy6m1unkmZ3bSLXGcXP/RpL9KhsvI11miSPNYTNKhpRHJM7b6XTyqzXLGvWT7K8o57a1K3j/pGD6SClKa2v55NABprgFbyMaU1yCcjy/eCeUHFOhk1+/3vLub1UV8P07sPbrOk+7is85XLP0dp1UPH8YWVFw1P2b6ZyYxO0Dh7TofD/r0o2PDwWn3WwtKeZvplQM7xALF37AckcSk7I70C05hZ8v+5786ipu9nrN3C0b2Hi8iIeGjmJ1UQErC46ysvAom4uPY29E670vdy2ZRnJiqSWOdEPbeK7zEN7oKPgyo8ltZ2KSyFVkixQDxjYvGqWZHKuu4qZVS1ljzKitJhMPDB1Ju7gETJvqJqEtPJLHvY4RzU70CojFqkplDD1Jhay+cq+7fITrVt6p1utGWM/k4GJRfh5zt6wH1Pt5ZuwkeqY07sCe1rEzT46ZyK2rl+EA/rZtI6lxVn7Rq1/dA12qfT1sLpt3YopqdhQBVhcWsLWk2Of+tUUFfHf4ICd37BLwu1qUn8ev1y6nzGZjgSs6xuHk5V3bubZvveS3tEwVqvry3ep7Wf6JipppTha306kq7n79eqMVewutCaTUeJnCZv88rCGTNQ4H921Y495+YOhIUluomZzSuSujM7NZW1TQYJ8VEw8PH83x2hqWHTvCqsJjbEjO4mB8Mjk1FYw5toebdmzhhV0WTPguwfDpoQN8duiAz/1Wk4nh7bK4uLaY8wr3AlBiiaPYEu8WCqA0sl+vWc6L46fQISGxRe872rQ9oRBBtpYc57qVSzhUqWbh7eLieXbsJCa2V81hDhh/QtcNeX9FOR/k7uOSHo3UUQo1Xfoo85PRctKGCSvOuvZCS8M/7c7SEn6zdrk7DPHBoaOY1N5/s5s5Xbrxt5Hj+P1PKnnrjxvXkWyxcoHLVFZaqEobe3HMmkB778S10bMjYt4A2FHqWyC4uG7VUsxAp8QkcpKSyUlOIScpmW5eS5vDwU2rlzbqiH9ky3p6pKRySv3IrM69lHY2by7ghM9egKwu7OzQi8flZn7vZd54a99uLuvR28smb3B4j8pq9yqlQmKqMjUW5aueHt5c+Ds1UQgjL+6S7CxT2uDMjl34WeeW98qyms28OmEqf9q0jk8O7q+z741JJ7t/lzf1G0iNw8GG44UcqdhPzrYlpNtrmVx6mEUZOQGDP7z3J5jNjMrMZnx2B8ZntWdUZrYqavnsbe5j7ug1iXtz19V5TbXDwfrjhZz343e8PP4kRHpGS99+QLaVFPOE3MzeclV36WBFBe/s38NF3Xs1/M00AS0UmsnXhw9y+9oVVBiZjf1S03hp/EmNzqgtJhNmlFP1qe1bOK9bz6CyPlvMwAmqoB5wOD6JbjX1bPY1larOU4YqN1BUU811KxdTajhbf9GrL1f0Cq73wwXde1Fus/HAJvVnufOnVaRYrcwx2eCtPyuTiReVZitgCIXew5TZL0JkBJk96wDyqirJq6p0+1aawhNyM8PbZZIVn1BX4xDjVHjo16+B04H9f49yuziFTXEp/N7r9fduWMOu0hLuH2r0U64sh4VvwaovvfIOTDDmFFVzy9VkZvql8C/DUOLSHsPI/vIyntq+BYBEs4U/DRvVopuSN+lxcfxz1HjuHTyCI1u/hGqojUtoMFGJN5tV2YzJZ8I2lfj5aFwtr/QVPL+rcY3YRVZ8PL/sM4AJ2R0YlpFJQv1Wpd/OU8IWYPAkfjfnGszPKJ+F1WTi2xlzuG7lYmRpCQcrK7hoyQKeHjOJkzs2rh2Hgo3Hi7h06SIqvJJIa50O/rB+NXvLy7hzUPP7QrdOv0EUcTqdPLNjKzetWuoWCNM7dub9k2b5NLFgMnFeN2XPPVhZwbv79zR+XKiZeKbb+eyqP3PM6lX/prQQnv8d7PyJWoeDX61Zzj7DyTqlfUfuH9K05u5X9u7HnQPVj9EBvP/d+9hevssjEDr2hGsegcnnehLGzGb4+QPK5h4hZnTsQqqfyJZki4VLuvdmWofO9E1NczvTm8rmkuNM/OZTBnz2PiO//IiZC77goiULuXnVUu5LyWFjH1VF1VJdwZNyARm2hhEzr+zZwbbiIpVN+9QtqnigSyDkDFBRaGfdXLfrWGJyxOpiOZ1O/rhxnVtbuk0MpltyCHM9DLISEni91ziWpXbk1Z5+zMPdBqgihUDHvRv4w4BBdA7gxzu/Wy9u7T+IsVntGwqEw3uVjxJUoMec6xiU3g6zO+HRRLfkFN6ZMpOTO3QCoNRm45crF/PW3l3NeatB8dDmn+oIBG+e3bmN3WXNbwWrNYUmUG23c9f61Xzkpcpe22cAdw8e7o699sWvBwzmo4P7sTud/HvHVi7s3qvhDzDUpGWpYmyfPQ+5qqZ+pdkKPfqq0MhjuWr534f4cdBUlid1BZOZXimpPD1mUrN8Hzf3H0iZrZbypR9z/4G1WFzKef8xyoyRkAQ9h8A6r8Yv4f4c6pFstTIgPZ21RYUN9pmAR0eM48wcT0NAp9NJYU0NuZXlHKyocC8/ObjfbykHb4praymurWVPuadk9zsZ/Xgj9QATy47Qu7qUZ3Yvdu8zO51cfnQHG5MySX/9fij05CCQnA6nXKl6LUe5tMJnebl8f1TlmIi0dK7tE0QRwWayPLsHbydm0cvX5AvU5zFokqowXFmGad9mrujZl3/KxiPjLCYTl/X0kX3tsKu8H1eU2Oxf+OxHkR4Xx8vjT+LBTet4c99u7E4n925cy57yMu4K4v7QFHIrylld2HhyoIsn5GZ+N3Ao3ZJTPAIsSLRQCJKjVVXcuHoJ64wbSZzJxMPDxwTtH+iZksoF3Xvxzv495FVVMm//bq7uHYJGI4HI7qL6QfzVyGY1W+CXf1Zx3J+/pHoZ4GTm1h94Nb0z9/ebxkvjpjS/QJndzu93L8N0wON0/G/nQYw47WaGRThM2BefHjrgFghxXn8YE/DahKkN1H6TyUR2QgLZCQmMaOe5KfRKSeWhzXVDH70RaemMyMymoLqKgupqCmqqKaiucmuYtWYLN/edygfbvqJ3dRmTS/MpMXv8Kpce28kjFUWeeC+TWWWuz7g0Os1k6lFSW8vDmzzv/5HhY8IXRNEUBk/ylJ3fsowbz7iRjcWFfH24bqdfi8nEoyPG0sdX7tDqr1S7T1Dl88ec6veyVrOZh4eNpndKGn/esh4n8NLu7eyvKOPxURNItobmdrsnCC1g/qEDzD90gCSLhf6p6Qxogo9DC4V6OJ1OVhcW0NErJG1zcRE3rFziDmHMio/nmbGTmZDdoUnn/nX/QXx4YK8K39yxjUt79CExUrNkL3UXUOaac25lR1YO3Rb+lySHnZNLDvO1/IqEkSMhrRlN0Ksq4L1/YDLamjow8WD3Mfyn4wAyVy7mf1Nm0L855w0hhysr60TJvDxhKtbNKkrMYjY3yQ58cY/evLlvF7sa+ZOmWeN4ZuzkRm84lTabEhA11RyqqOCbvM1cdXQ7CU4H6Uamd7eaMizeSkiPwXD6dY03e4oS/9y2iSPVKrrtkh69Q1MKOxT0GKgSPcuLYety4s64gWfHTmZBfp47ItBsMvHV9J/5TiYtKYBvjT4NZguceVNQWpnJZOLavgPonpLCb9euoNJu5+vDh7h06SJeHD+FTi0ISd9SfJyXd29nfj2nuz8q7XY2FBexobiIYKd5MSDWY4cdpSWc9v03XLx0oaf4lsPB+YsXuAWCSEvno6mzmywQALolp3CxoVkcra7izTDaHINhb3kZF5XD+eJU9iSoP0dC+XF47T6VwxBM6QoXx4/AK3d7+lzHJ+G8/F6OjlBNT4pqa/j5su/ZX15mZD63vE/Tgvw8Ll26KGC9KRcOp5M716+iuFbdeK/u3Y+pHTrR3B5gKVYr8yZNZ07nnDp/JBPw9uTpPmegSVYr3ZJTGNEui9PSUrjhyDbi6xWsc00VjlgT+VfvCRRcfn9MCYQNxwv5z15lksyKj+euQcN9HltpRLlVNhLtFhbMFlU/DJR5dN9WzCYTszt3xWLc2M0mk//qAp+/6GnFOuW8JlckPrVzDv+bPIOORnjqxuIizvvxO7YUB6hnVg+H08m3hw9x2dJFnPHDN3yQuw9bgP9lktnC/w0cysU9ejMqM4vUJmooWigYHK+p4Ypl3yMbCVesMZxoszp14b2TZtK9BY60W/oPckcePbtzGxXhKKsQBCW1tVy/cjHFtbVsS87k1dk34Bxs1I9x2OGrV+CdvweXZZy7XSVlHTFmMBkd4Nq5WAaM4V+jJjCtg5p9H6mu4pwfv2PKt59hc6gftt3haLZT7HG5iRUFR3nch724Pv/Zu4sfj6ookn6pafzBz40sWDokJvLsuMksP+Ust93YYjYzONhEJqONY32x5ATeaN+fWUPP4omsvpy3ZAE7SiNbSMDXzdzmcHDvhjVusX7P4BF+zY2v9hwb2EEcagZ7FcTbsrRpr926QvU1AcjqAidf2KwhDGuXyYdTZzHQMN3kVVVy8ZKFLDQmMbkV5T5Lw1TYbLyxZyezFn7J9auWsLzgqHtfTlIyvxNDGdbIbyzZYuHF8VO4pf8gHh0xlg9OmsWGOeeyeNYZQY9bCwWDt/fv5mi175K7OUnJPD9uSosTcromJbsdWwU11bxhzLYiid3p5La1y91x5WOz2nPv6EmYLvo/1XTcFW2zdRm88HsVgeGm3o940xJVZ8nVWjRngKrJZMysEiwWnh07iXGGaeF4bQ15XpnETuD8xd+xsxk3vDJDoJYFIVh3lZbUSch7fNSEkJruOiQmNi8MM6MDxHmSnQqNAmv5cUlMLzlEJyOM+EBFORcsXsBiQ6hFAl838//s3cUmY8Y7MbsD53fzP4tent2Dy8Vslmf3CNtYG9BzCCQZmsDW5cGXl6+qUFqCizNvalFkXNekZN6dMoMZhlmy3G7jupWLmb3wS6Z+97k7c9rmcLC5uIhDlRX8dcsGJn37KQ9sWufOQQAYk5nNv8dMYtHM0/jVgEG8d9Is/jVqAsnG77hdXDzfzTiNKUYUlAuTyUROcvCVArRQMAj0ZztcVRmaRuPALf0GuovJvbBTUmYLvjhXc7A5HO4ZidPpZO7m9Sw6oiJGcpKSeXbsJBUJZTKpMNZrHvH0FS7Mg5f+oEov/+dP7o5T2G1Kk3jvH6qWEqhGI1c/1KClZpLVym/6+27yUlxby1+3bgjtm/aixuHg9nUr3WGTvxVDWtRXOqTEJ9bpK1xuzMprTWZ61JTzwe4fmJKpnNultlquXvEjb+3bHZGhNXYzP1xZyWOGZhZnMvHIsNEhy0kIKRaLytMBlfHtI3u/AQveUi1xQUV39Wm5NplqjeOFcVO4qrfK8ndAo36oc3/8jqnffc7zuyQlhonTYjJxVtfufHjSLN47aSand+2G1bh3xJvNnNOth7t0Trv4eDontTyYI2xCQQjxihDiiBCiUd1eCDFdCFEshPjJePjr9Nam6JiYxM+NMhBFtTW8tjt82sLyY0eY5jUjsTudvLxHFcJzqZrt66fldx8INz4GfY08BVuNahKz6ydMhqaQZauuq5affBFccIfPWdU3+Ycafd7Fwvw89x8h1Dy1fQsbi1UpiDGZ2dzYV4TlOs1m9i8a74xmjSf9gt/y6uQZXNZDaZd2p5N7N6zhz5vXN1qfJ9w8tPknt2Z2U7+B9I1y4IBfBgdXTttN7naVBwJKyzj16pANxWo28+DQUczq6Ltons3pdPsy06xx3NhX8MOs03lyzERGZjYeChsOwqkpvAbMCXDMj1LKkcbjoTCOJSAn1VO56jO5fccmx/v648Z+giRD7XtxtwzLDXFXaQm/XLnYZ4OcPwwazqB0H7bvlHS44j5PLwCDjkYtpRSHl8lmzrUw83K/0RnFAeL5HUBJkDH/TWFN4TGe2aHKQaRYrDw2arx7phUzJCTBVQ+pFqHeXeB+/W/oP5o4s5k/Dx/NvYNHuLXVl3Zv56ZVS8NT6tsHC/Pz+CIvF4CeySnc2j8MzYJCSe9hKuEMlFDwJ0TtNpWT4DKPzvll3YTAEFHoo6S3CzMmHhw6imWnnMldg4fTNSn8BSIbjiFMSCl/ABpmB8Uol/bo475J18diMoX8D9AhIZErDW2hpLaWV3dvD+n5Qd04Ku2+yzNvD1QDyGxpUFywfpQMEFT2bL9U/3+wZIsl5BVky2027li30l3H6f6hI+nhL/EpmlgsRp9f4zdoNrvLj4CyC1/XdwDPjZvs/p1+m3+Ii5csJK8y/CXHK202/rhxrXv74eFjwp982VKscZ6OfiXH/JaKZ9knkL9XrfceDsOnhWVINXb/vo1ki5mrevcjJUQ5Dc0h2lOmSUKI9UKIL4QQLauz20JyK8qp9rqBlhllEMotcTw9ZmKzQlADcUNf4W4k8vLu7RwPpitbE1hyzH+Hr0D7AdUVrRFqvT0sFZ7eNgQAABPoSURBVIGjhy7u0cvtRwHP5+taVtjtvLRL+m/U00Qe2fwT+42yHad06srFTellEaOc2jmHdybPoFOiMvltKTnOeYu/Y1Og3t8t5KkdW8k1hM/ZOd2NUN5WQJ0oJB8mpMLDsOhttW6NV87lMPlJxgTI5RgXhvtMU4mmUFgL9JRSjgCeAj6K1kBsDgf3bFjjnlHe3HcgL/cYw7LUjrzZezxzunQLy3WzEhK4uo/Kai612XgpDNpCi2mfg3fAZLHhCD3mFTFDx8BRJR0Tk3hyzER3OO7jXYezLLUjj3f1OPL+vm0Td6xbWUc4N5dvDh/ibaPGVHZ8An8ZMSY2HaLNYGi7TD46aTZDDNNfflUVFy9dyNeHVSmMUOWBuKhxOHjRKCqXZo3jvsFNq4kVVfqMUM58UNF09T8Xp1OVgXEFS0y7WFUBCBNX9e5XZ3LkjQm4Pgb8XVETClLKEillmbH+ORAnhIhKSuQbe3e6HZHjs9rz+0FDWd+xT0TC6K7rM4A0Q1V8bfcOCqv92xybwoAATsBAfhRAmTAGTXBvxhnmowyj2Qjp2arqZxCc2jmH72bM4ZZ+A1mepSJbNnfsy19HjHWXm/jo4H4uW7aIo1W+w4MDcay6irvXe3rwPjpybENnuhcRT64KAZ2TkvjflBnM7qRKc1fa7dy0aim3rl7G7IVf1skDWX+8aVbcwupq7t+wln1GOOShygp3wtSdg4bRIbEV9QuIi4cBxu+zKJ/4+qGpG3/0dGnr2CPszbn6pKbx/LgpZMQ1/K3NHTE2YJn6SBA1oSCE6CyEMBnr442xNL0+cQs5WFHBP402kvFmM38ZPiakDuVAtIuPdxcRK7fbeKGRMr+uWjkVTZhBf5mXyw9+wmxTrNbgi5edeRN06gWgasu7lklpcMldTeqD0C05hf8bNMyd7p9stXJJj968OWk6WUYC1LqiQs798Vs2FzfdJOJ0Orl7/Rp3X+nLevRhVqeufl8TleSqEJBitfLcuMlcZ3yPTuDzvFx2e8W2O4GLlyxkVYH/AmouSmpruGTpQv67b1cDXSPJYnFX+21VeEUh1enNXFEKX75sbJhUtdkAfaFDwbSOnVk6+0z+MXKc+15jNZsi02clCMIZkjoPWKZWRa4Q4lohxE1CiJuMQy4ENgkh1gNPApdKKVus8zal9IHT6eSPm9a6b7a3hDjEzlxv6Ytr+gwg3Zg5vLF3Z4MkuiLjBudaBuLNvbu4dfUyao1ZUf0G6D2TU3h9wlT/1Sa9SclQJZrPu82jiienwa+ehpx+/l8bJOOy2/Px1NkI4/M/VFXJRUsW8qUR7RIs/9u/h2+N8NeeySncO2REwNdEJbkqRFhMJu4dMoLfDxzq85gah4MHNq6l0majym73+3hhp6dZTn0q7XbePRChsu+hpN9od6h0HaHwzeuqDAbAuJ+pUOwIkWxVTag8E9DYMW2GTSxKKf12TZFSPg08HerrPi43san4OOW2WmYGaPz9Rd5Bt/Dom5rGTf1C+6PIjE+ASmPph/S4OK7vI/in3ESl3c7zOyX3ed3MHPWWvnA6nTy5fQv/MhqegBJ0t4sh7Nz8GVRDtTWeBTNPa7o2ZI2DEdNVP+DCPNXpK8Qhe92SU3jvpJn8du0KvsvPo9Ju5+bVy7hDDOFX/QcF9AnsKy/jYaNqqRl4bNSEqEZxRJJA3+bW0mIGf/Fhi6/zycEDkanuG0riE6D/aNiyzG3+TLDbYN13an9qJsz6eRQHGFtEO/oo5ARb+qCktoYHN3la6v05DCF2ycmpdZb+uLpPPzLjlPnkv3t3ku8jt8AXdqeT+zeurSMQ/jhkJP83aBhWs9ndoOT1XuMiah5rKqnWOJ4fN4Wb+nkcbo/Jzdy2dgVVfsxnNoeDO9atdGt9t/YfxOis7LCPN1YoDlPiX8PrhD6XJCLUSw7M9O5CePr1qkd4E2iNfqhgOTGmUY3wt60b3WaaS3r0DkvIKTMuU12bgnBepVrjuKGf4NGtG6l2OHh25zYeHDoqqMtU2+38du0KvjSiT+JMJv45ajxn5XjMIUE1KIkRLCYTfxg0nH6p6dyzYQ01DgefHDrAvooyXhjXePnh53ZKd4P3YRmZ/HqA77IabZFAQQUmYFanrgHbwC4vOEqhHzNltEufN4uifPjh3TpPxbk8JqntQIxv8ilf7TmWs/etZX7P0cwNxRhjiBNSKKwuPMabRv2Y7PgE7g5BtcxGGTC2QfKXP67s1Y+Xdm2noKaaeft2c2NfQZcAGY0ltbXc6FVFMcVi5dlxk1tPHLkfLujei14pqdy4aikFNdVsOF7EOT9+ywvjptAvNd2dzVtaW8sT2zcDqkfw46PGR6XZSzRnj6d36cbcLRvcDvb6nNutJ4+NCnzz++JQLres8V0SwpVw2apY8JbHd1CfsuOwa12T/qfQuiZZTaXNmY8CUeNwcM96T5OVB4aODLqRe7hJtlq52fBr1Dgc/Nsoz+DrZnOkqpJLly50C4Ts+ATemjytTQgEF2Oy2vPR1FkMMsoP51dVceHihYz5ar5b0yuoqXaHTN49eHjU6vFEM4opyWrl+XGTSW8kEmxku6ygtc45XXJ81ob6w6BhzQqZdNXzb2pd/5BgtwUunb3xx8iMpZVwwgmFF3ZKdpSpWcO0Dp05s2v3AK+ILFf06utuzPHO/j3kVpQ3erPZU1bKhUsWsrVElaronpzCu1NmMLxd5ApnRYpuySm8O2Ump3ZWoaW1TgdVjob+hTiTiTO7hifRMBiiHcU0Jqs9C2aexp0Dh7kdzxaTiXemzHBHtwXCZDJx1+DhfHjSLNIMAZNujePzaac0OxDjdjGUidkduF34jpAKG7U1SjAYuII1qk1et76q8siOKcY5oYTCnrJSntqhHLGJZgsPD4+9sr+JFgu3GHWWap1Ont6xtcHNZuPxIi5aspADRgmHQekZvDdlBr39dZJq5aRYrTw7drLfAmG1TifvHNgbuUHFINkJCdzcf6C7w5jJZGqWKW1kZhbZCSpqLishwXfhxCCY2akL8yZPDxgNGBYSklTPCoMCa2KdJRBURv6JxAkjFJxGyWFXF7XbBw5pUQe1cHJpj950MZyp7+7f666JZHc6+fFoPpctXeS2HU/I7sDbk2eEvJhcLOJwOgMWf1saTD0nzYmDyQTjTnNvVhkFB+2uyaDZAmNOicbIYpYTRih8kLuPZYbtfXB6O34Zw7HWCRYL1xg1kRw4OW6EAR6oKOfq5T9QbqjDc7rk8PqEqUGbBlo7JpMpYDy+OYaSgDQxwqSzYehJDZ83W1RCZlYUNJgY5oQQCgXV1TyyWbViNAN/GTEmYE39aDrHnE4n3xxuvCmNyyZ6Rc8+PD1mUuyXLw4hFpOpQavB+pzcse042TUhwmJRDaB+8WDDfhXDpkZ1aLHICSEU/rxlvXu2fVXv/owIwhkbTefY6sICVhX6rlUTbzZz96Dh7kbxJxK/6j/I5/vumpTMhd1jo36MJsYwmaDviLr9KjL1BKIx2rxQWHw0nw9z9wHQJTGJOwYG17Yhms6xZQX+7eI1DgebS45HaDSxxfjsDjwzdpK7n4CLke2yeGvStBPGlKbRhIs2LRSq7Hbu2/D/7d1/kFV1Gcfx9112QXYxwAEEF0RI5kFCWNFRJsoKxAFh9K8msHRCq5nCSqemoZ/M9EfTH00DaukoGkGEFmBaOaQzYdREZpBJSE8hJCwi0DARRCA/tj/Oucfbtj/u3j13v+ee/bxmdnbv3mXuw7k/nu/P5/v2noSvXz2DIT2o6BlKOe3/LJeqqLabRzfz6zkLkp3Nlw1u5Kn3zmF8DjYSlVtEsTt5LsPQG2ld3zzL9bV54K+v8nq8bHPemGZuGt11CeWseN+o0V3eP7xhIFcPHd5H0WRTQ11dcixld6UbuhN0c1U7xeKJ3RVR7E6tlgOvtrSub1rSSN5pv35zmxTOXriQnE1wcX192Ts6s2DasEuYParzYatPTprcryaYqy3o5qp2GuPntbGXz2/ojXRZldb1TUsayTvt12/4plGK9p08wYm4WuTRM6dLToua1mERtSy7/9qZfPmV7fz04P5kxVEB+PzkqcmhKpKO2ZeOCbOxSnokSz26tKRRQynt128uegqn4yqhs7dsTjZ1FTeptQy7hNvHTwwZXkWa6utZMeMGts5ZwMi47MXljU18qoxzBUTyKEs9ujzLRcr96s4dPH1wf4f3DR84sKYnZZsbG2mqr+foGZQMpF9Tj65v1HxP4eCpU2zqot7NC0feZH/JmbUiIlmRxSGxmk8KLx072uUxlW3Ai3F5C0nBoMH/+11EKpbFIbHspKcKlbOrt5aHjzKnB6fJiUjXsjgkVvNJYeaIUTQUCpyNVxq1V18o8J4cHToTXA9Pk+tPUhkKUE9MAqtaUjCzx4GFwBF3/7++kZkVgJXALcAp4KPuvqOnjzNy0EXcOeFKHtv7tw7vv338O2tuOarUpvtsKo++5ny8k5PLyqKeWE3I4lxAWqr5P1oNPAis6eT++cCk+OsG4KH4e48tu2oadYUCa/bt4cyFt2cY7p44iWXVOn9ZpJ1UhgLUE6sJqTQAMqpqE83uvhU41sWf3Aascfc2d/8dMMzMKnpH1dfV8aUp09k2d2FylOXljU185V0t3ZbIrgVptEry3LIR6WtBT5OrspCfmM3AgZLbrfHvKjZ84CAa4w+9PE0up7FCIYurHKR61AjohOZsuhXyFdPRp3bHs8X9XBrDEllc5SDVk+fhjV7RnE23QiaFVmBcye2xQMfHjYl0QK3hzqXRCMjl9dWcTbdCDh89A9xpZgUzmwkcd/dDAeORPpTGB46GxKpL17d/quaS1PXA+4ERZtYKLAcaANz9YeBZouWoe4iWpC6pViySPWkMb2hIrLp0ffunqiUFd1/czf1twNJqPb5kmz5wRLKp9tdriohIapQUREQkoaQgIiIJJQUREUkoKYiISEJJQUREEkoKIiKSUFIQEZGEkoKIiCRylxRyWcRLRKSP5C4pqIiXiEjlctecVk0dEZHK5a6nICIilVNSEBGRhJKCiIgklBRERCShpCAiIgklBRERSSgpiIhIoub2KezatesfZvZ66DhERGrM+HL+qNDW1lbtQEREpEZo+EhERBJKCiIiklBSEBGRhJKCiIgklBRERCShpCAiIoma26fQHTObB6wEBgCr3P2bgeJ4HFgIHHH3oCf+mNk4YA0wGrgAPOLuKwPFchGwFRhE9Prb4O7LQ8RSEtMA4A/AQXdfGDiWvwMngPPAOXe/LmAsw4BVwFSgDbjL3bcFiMOAJ0t+NRH4mruv6OtY4njuAz5GdE12Akvc/XSIWKohVz2F+M39HWA+MAVYbGZTAoWzGpgX6LHbOwd8zt2vAmYCSwNelzPAbHefDrQA88xsZqBYij4L7A4cQ6kPuHtLyIQQWwlsdvfJwHQCXSOPtLh7C3AtcAp4KkQsZtYMfAa4Lm7sDQAWhYilWnKVFIDrgT3uvtfd3wKeAG4LEYi7bwWOhXjs9tz9kLvviH8+QfTmbg4US5u7n4xvNsRfwXZQmtlYYAFRi1hiZvYO4EbgMQB3f8vd/xk2KgDmAK+5e8iqBvXAYDOrBxqBNwLGkrq8JYVm4EDJ7VYCffhllZldAVwDvBgwhgFm9jJwBHje3YPFAqwAvkA0rJYFbcBzZrbdzD4RMI6JwFHge2b2RzNbZWZNAeMpWgSsD/Xg7n4Q+BawHzgEHHf350LFUw15SwqFDn6nOh4xMxsCbATudfd/hYrD3c/HQwFjgevNLMici5kV53y2h3j8Tsxy9xlEQ6BLzezGQHHUAzOAh9z9GuDfwLJAsQBgZgOBW4EfB4xhONHowwTgMqDJzD4SKp5qyFtSaAXGldweS866dpUyswaihLDO3TeFjgcgHo54gXBzL7OAW+PJ3SeA2Wb2g0CxAODub8TfjxCNm18fKJRWoLWkF7eBKEmENB/Y4e6HA8ZwE7DP3Y+6+1lgE/DugPGkLm9J4SVgkplNiFsVi4BnAscUnJkViMaGd7v7twPHMjJe1YKZDSZ6k/0lRCzu/kV3H+vuVxC9Vn7p7sFafWbWZGYXF38Gbgb+HCIWd38TOBCv/IFoLP/VELGUWEzAoaPYfmCmmTXG76s5ZGuRQq/lKim4+zngHuAXRE/Uj9x9V4hYzGw9sC360VrN7O4QccRmAXcQtYRfjr9uCRTLGGCLmb1ClMSfd/efBYolay4FfmNmfwJ+D/zc3TcHjOfTwLr4uWoBvhEqEDNrBOYStcyDiXtOG4AdRMtR64BHQsaUNpXOFhGRRK56CiIi0jtKCiIiklBSEBGRhJKCiIgklBRERCSRuyqpItViZueJliE2EBUZ/D6wwt2zUiJDpNeUFETK95+4PAdmNgr4ITAUCFr6WyRN2qcgUiYzO+nuQ0puTyTagDcCGA+sBYpF4+5x99+a2VqiMyOejv/NOuBJd+/3O+0lmzSnIFIhd99L9B4aRVTxdW5czO5DwP3xn60ClgCY2VCiOjnP9n20IuVRUhDpnWJl3gbgUTPbSVTFcwqAu/8KuDIebloMbIzLsYhkkuYURCoUDx+dJ+olLAcOE51QVgeUHs+4FvgwUdG9u/o4TJEeUU9BpAJmNhJ4GHjQ3duIJpwPxSuR7iA6prFoNXAvQKgCjSLlUk9BpHyD4xPjiktS1wLFUuTfBTaa2QeBLUSH0gDg7ofNbDfwkz6OV6THtPpIpMriss87gRnufjx0PCJd0fCRSBWZWfEQoQeUEKQWqKcgIiIJ9RRERCShpCAiIgklBRERSSgpiIhIQklBREQS/wVWN8C4tyG21AAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plot_itpt_hpm()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Analysing HPM gain from baseline to BMI across sessions for IT vs PT mice"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 112,
   "metadata": {
    "code_folding": [
     0
    ]
   },
   "outputs": [],
   "source": [
    "def plot_itpt_hpm():\n",
    "    \"\"\"\n",
    "    Aggregates hits per minute across all IT and PT animals. \n",
    "    Looks at max hpm in 10 minute windows.\n",
    "    \"\"\"\n",
    "\n",
    "    # Getting all hits per minute arrays\n",
    "    IT_train = []\n",
    "    IT_target = []\n",
    "    PT_train = []\n",
    "    PT_target = []\n",
    "    num_it = 0\n",
    "    num_pt = 0\n",
    "    bin_size = 10\n",
    "    \n",
    "    for animaldir in os.listdir(datadir):\n",
    "        animal_path = datadir + animaldir + '/'\n",
    "        if not os.path.isdir(animal_path):\n",
    "            continue\n",
    "        if animaldir.startswith(\"IT\"):\n",
    "            num_it += 1\n",
    "        else:\n",
    "            num_pt += 1\n",
    "        animal_path_files = os.listdir(animal_path)\n",
    "        animal_path_files.sort()\n",
    "        session_idx = 0\n",
    "        \n",
    "        for file_name in animal_path_files:\n",
    "            result = re.search(pattern, file_name)\n",
    "            if not result:\n",
    "                continue\n",
    "            experiment_type = result.group(1)\n",
    "            experiment_animal = result.group(2)\n",
    "            experiment_date = result.group(3)\n",
    "            f = h5py.File(animal_path + file_name, 'r')\n",
    "            com_cm = np.array(f['com_cm'])\n",
    "            try:\n",
    "                xs, hpm, _, _ =\\\n",
    "                    learning_params(\n",
    "                        experiment_type + experiment_animal,\n",
    "                        experiment_date,\n",
    "                        bin_size=1\n",
    "                        )\n",
    "            except:\n",
    "                continue            \n",
    "            # Get running mean over 10-minute windows\n",
    "            hpm_5min = np.convolve(hpm, np.ones((5,))/5, mode='valid')\n",
    "            max_hpm = np.max(hpm_5min)\n",
    "            hpm_gain = max_hpm - np.mean(hpm[:5])\n",
    "            if experiment_type == 'IT':\n",
    "                IT_train.append(session_idx)\n",
    "                IT_target.append(hpm_gain)\n",
    "            else:\n",
    "                PT_train.append(session_idx)\n",
    "                PT_target.append(hpm_gain)\n",
    "            session_idx += 1\n",
    "\n",
    "    # Collect data\n",
    "    IT_train = np.array(IT_train).squeeze()\n",
    "    IT_target = np.array(IT_target)\n",
    "    PT_train = np.array(PT_train).squeeze()\n",
    "    PT_target = np.array(PT_target)\n",
    "    fig = plt.figure()\n",
    "    ax = fig.add_subplot(111)\n",
    "\n",
    "    # Some options:\n",
    "    # Order 1, Order 2, Logx True\n",
    "    sns.pointplot(\n",
    "        IT_train, IT_target,\n",
    "        color='lightseagreen', label='IT (%d Animals)'%num_it\n",
    "        )\n",
    "    sns.pointplot(\n",
    "        PT_train, PT_target,\n",
    "        color='coral', label='PT (%d Animals)'%num_pt\n",
    "        )\n",
    "    ax.set_ylabel('Number of Hits')\n",
    "    ax.set_xlabel('Day')\n",
    "    plt.title('Gain in HPM from Experiment Beginning')\n",
    "    plt.legend()\n",
    "    plt.xticks(np.arange(0,18,2))\n",
    "    plt.show(block=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 113,
   "metadata": {
    "code_folding": []
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "No handles with labels found to put in legend.\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plot_itpt_hpm()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Analysing % Correct across sessions for IT vs PT mice"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "metadata": {
    "code_folding": [
     0
    ]
   },
   "outputs": [],
   "source": [
    "def plot_itpt_hpm(bin_size=1, plotting_bin_size=10):\n",
    "    \"\"\"\n",
    "    Aggregates hits per minute across all IT and PT animals. Performs regression\n",
    "    on the resulting data, and returns the p-value of how different linear\n",
    "    regression between the two animals are.\n",
    "    \"\"\"\n",
    "\n",
    "    # Getting all hits per minute arrays\n",
    "    IT_train = []\n",
    "    IT_target = []\n",
    "    PT_train = []\n",
    "    PT_target = []\n",
    "    num_it = 0\n",
    "    num_pt = 0\n",
    "    \n",
    "    for animaldir in os.listdir(datadir):\n",
    "        animal_path = datadir + animaldir + '/'\n",
    "        if not os.path.isdir(animal_path):\n",
    "            continue\n",
    "        if animaldir.startswith(\"IT\"):\n",
    "            num_it += 1\n",
    "        else:\n",
    "            num_pt += 1\n",
    "        animal_path_files = os.listdir(animal_path)\n",
    "        animal_path_files.sort()\n",
    "        session_idx = 0\n",
    "        \n",
    "        for file_name in animal_path_files:\n",
    "            result = re.search(pattern, file_name)\n",
    "            if not result:\n",
    "                continue\n",
    "            experiment_type = result.group(1)\n",
    "            experiment_animal = result.group(2)\n",
    "            experiment_date = result.group(3)\n",
    "            f = h5py.File(animal_path + file_name, 'r')\n",
    "            com_cm = np.array(f['com_cm'])\n",
    "            _, _, perc, _ =\\\n",
    "                learning_params(\n",
    "                    experiment_type + experiment_animal,\n",
    "                    experiment_date,\n",
    "                    bin_size=bin_size\n",
    "                    )\n",
    "            if experiment_type == 'IT':\n",
    "                IT_train.append(session_idx)\n",
    "                IT_target.append(perc)\n",
    "            else:\n",
    "                PT_train.append(session_idx)\n",
    "                PT_target.append(perc)\n",
    "            session_idx += 1\n",
    "\n",
    "    # Collect data\n",
    "    IT_train = np.array(IT_train).squeeze()\n",
    "    IT_target = np.array(IT_target)\n",
    "    PT_train = np.array(PT_train).squeeze()\n",
    "    PT_target = np.array(PT_target)\n",
    "    fig = plt.figure()\n",
    "    ax = fig.add_subplot(111)\n",
    "\n",
    "    # p-val for linear regression slope similarity\n",
    "    p_val = linreg_pval(IT_train, IT_target, PT_train, PT_target)\n",
    "    print(\"Comparing linear regression slopes of IT and PT:\")\n",
    "    print(\"p-val = \" + str(p_val))\n",
    "\n",
    "    # Some options:\n",
    "    # Order 1, Order 2, Logx True\n",
    "    sns.pointplot(\n",
    "        IT_train, IT_target,\n",
    "        x_bins=plotting_bin_size,\n",
    "        color='lightseagreen', label='IT (%d Animals)'%num_it\n",
    "        )\n",
    "    sns.pointplot(\n",
    "        PT_train, PT_target,\n",
    "        x_bins=plotting_bin_size,\n",
    "        color='coral', label='PT (%d Animals)'%num_pt\n",
    "        )\n",
    "    ax.set_ylabel('Number of Hits')\n",
    "    ax.set_xlabel('Day')\n",
    "    plt.title('Percentage Correct Across Sessions')\n",
    "    plt.legend()\n",
    "    plt.xticks(np.arange(0,18,2))\n",
    "    plt.show(block=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "metadata": {
    "code_folding": [
     0
    ]
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Comparing linear regression slopes of IT and PT:\n",
      "p-val = [0.40831113]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "No handles with labels found to put in legend.\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plot_itpt_hpm(\n",
    "    bin_size=1, plotting_bin_size=10\n",
    "    )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
