import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.colors
import mpl_toolkits.axes_grid1

############### Deprecated API ###############
def compare_img_seq(imgs: list, x_titles: list=None,
                    y_title: str=None, cmap=None, normalize=True):
    import numpy as np
    assert type(imgs) is list
    all_imgs = np.stack(imgs)
    if all_imgs.min()<0 or all_imgs.max()>1 and normalize: # normalize if needed
        print('normalizing')
        all_imgs = (all_imgs-all_imgs.min())/all_imgs.max()
        imgs = list(all_imgs)
    fig, _ = plt.subplots(1,len(imgs))
    fig.set_size_inches(len(imgs),1)
    for i in range(len(imgs)):
        plt.subplot(1,len(imgs),i+1)
        plt.imshow(imgs[i], cmap=cmap)
        plt.xticks([], [])
        plt.yticks([], [])
        if i==0 and y_title: plt.ylabel(y_title)
        if x_titles and x_titles[i]:
            plt.title(x_titles[i])
        #plt.colorbar()
    #plt.tight_layout()
    plt.show()

def display_3d(sol, y_title=None, x_title_func=lambda t: f't={t}',
               img_getter=lambda sol, t: sol[:,:,t],
               time_samples: list=None, cmap=None):
    if x_title_func is None: x_title_func=lambda t: ''
    img_seq = []
    title_seq = []
    if not time_samples:
        time_samples = list(range(1,sol.shape[-1],max(1, int(0.5+sol.shape[-1]/10))))
    for t in time_samples:
        try:
            img_seq.append(img_getter(sol, t))
            title_seq.append(x_title_func(t))
        except Exception as e:
            print('Error: ', e)
            break
    compare_img_seq(img_seq, x_titles=title_seq, y_title=y_title, cmap=cmap)

# Verified to work: 5/10/24
def display3d_dataset(data_loader, n_sample_data=3):
    batch = next(iter(data_loader))
    print('batch["x"].shape=', batch['x'].shape)
    print('batch["y"].shape=', batch['y'].shape, '\n')

    import random
    for i, datum in enumerate(random.choices(data_loader.dataset, k=n_sample_data)):
        print('-'*50)
        print(f'| Sample datum i={i}:') # NOTE: we skip the 3 positional embedding channels
        print('-'*50)
        x = datum['x'].permute([3,1,2,0])[0,:,:,:-3]
        display_3d(x, y_title='input')
        display_3d(datum['y'][0], y_title='output', x_title_func=lambda t: f't={t+x.shape[-1]}')
##############################################

import os, re

'''
def proportional_allocation(scalar_allocation, proportional_to_size, int_cast=True):
    """
    Dynamically allocates a quantity across dimensions proportionally to the size of those dimensions.
    Such that prod(new_size)==prod([scalar_allocation]*len(proportional_to_size)).

    This is equivalent to resizing a hyper-cube with side_length=scalar_allocation,
    to be proportional to proportional_to_size while retaining the same volume.
    """
    c=((scalar_allocation**len(proportional_to_size))/np.prod(proportional_to_size))**(1/len(proportional_to_size))
    new_size = np.asarray(proportional_to_size)*c
    return list((new_size+0.5).astype(int) if int_cast else new_size)
'''; # does same kind of dynamic allocation as for modes

default_img_scale=1.45

class GridFigure: # Verified to work 5/18/24
    def __init__(self, title:str='', img_scale: float=default_img_scale,
                 cmap:str=None, y_title_vertical=True):
        self._img_scale = img_scale
        #if row_size: assert len(row_size)==2
        self._cmap = cmap
        self._rows = []
        self._last_x_titles = None
        self._n_unique_x_titles = 0
        self._title = title
        self._y_title_vertical = y_title_vertical

    @property
    def nrows(self):
        return len(self._rows)

    @property
    def ncols(self):
        try: return len(self._rows[0]['imgs'])
        except (IndexError, KeyError): return 0

    # adapted from compare_img_seq
    def add_img_seq_row(self, imgs, x_titles: list=None, y_title: str=None):
        assert len(imgs)==len(x_titles)
        assert type(imgs) is list

        if x_titles==self._last_x_titles: x_titles=None
        else:
            self._n_unique_x_titles+=1
            self._last_x_titles=x_titles
        row = locals().copy()
        del row['self']
        self._rows.append(row)

        assert len(imgs)==self.ncols # ensure consistent row lengths
        assert all([self._rows[0]['imgs'][0].shape==img_i.shape for img_i in imgs])
        # ensure consistent image shapes

    # adapted from display_3d
    def add_3d_row(self, array_3d, y_title=None, x_title_func=lambda t: f'{t=}',
                   time_samples: list=None, img_getter=lambda array_3d, t: array_3d[:,:,t]):
        if x_title_func is None: x_title_func=lambda t: ''
        assert len(array_3d.shape)==3
        img_seq = []
        title_seq = []
        if not time_samples:
            if array_3d.shape[-1]>10:
                time_samples = list(np.linspace(1,array_3d.shape[-1], num=10, dtype=int, endpoint=False))
            else: time_samples = list(range(1,array_3d.shape[-1]))

        for t in time_samples:
            try:
                img_seq.append(img_getter(array_3d, t))
                title_seq.append(x_title_func(t))
            except Exception as e:
                print('Error: ', e)
                break
        self.add_img_seq_row(img_seq, x_titles=title_seq, y_title=y_title)

    @property
    def _img_values_range(self):
        # initial extreme values
        min_ = float('Inf')
        max_ = -min_
        for row in self._rows:
            for img in row['imgs']:
                # update
                min_ = min(min_, img.min())
                max_ = max(max_, img.max())
        #print(f'img value range: [{min_}, {max_}]') # shown in colorbar
        return min_, max_

    @property
    def _row_size(self):
        # auto-size figure based on aspect ratio
        row_size=[0, self._img_scale] # row_size=(width, height) <-- confirmed
        img0 = self._rows[0]['imgs'][0] # img.shape=(height, width) <-- confirmed
        aspect_ratio = img0.shape[1]/img0.shape[0]
        #print('aspect_ratio:',aspect_ratio)

        # This little trick safely applies the aspect ratio to *BOTH* the height & width of a subplot
        # B/c you get new_aspect_ratio = (self._row_size[1] / aspect_ratio**0.5) /
        # (self._row_size[0] * aspect_ratio**0.5) = self._row_size[1]/(self._row_size[0]*aspect_ratio)
        # BUT it also shrinks the row height!
        row_size[1]=(row_size[1]/aspect_ratio**0.5)*(10/self.ncols) # height, we scale it based on the number of columns
        row_size[0]=row_size[1]*self.ncols*aspect_ratio**0.5 # width
        return row_size

    def show(self, fig_path: str=None): # render and possibly save the figure
        if fig_path: assert fig_path.endswith('.png')
        elif self._title:
            fn = re.sub('[^A-z0-9. =]', '', self._title).replace(' ', '_')
            fig_path=f'./grid_figures/{fn}.png'
            import os # make grid_figures directory if needed
            os.system(f'mkdir ./grid_figures 2> /dev/null')

        cbar_size = 0.15 # Set padding sizes
        axes_pad=(0.05,0.3 if self._n_unique_x_titles>1 else 0.05) # extra space for row sub-titles
        # NOTE: it apparently works better to make the padding independent of self._row_size...

        # Set up figure and image grid
        # NOTE: idea here is to add padding to figure size <-- works well!
        fig_size = (self._row_size[0]+axes_pad[0]*self.ncols+cbar_size,(self._row_size[1]+axes_pad[1])*self.nrows)
        #fig_size = (self._row_size[0],(self._row_size[1])*self.nrows)
        #print('fig_size_inches: ', fig_size)
        fig = plt.figure(figsize=fig_size) # fig_size=(x,y)=(width, height)
        if self._title:
            plt.title(self._title, pad=30)
            plt.axis('off')

        grid = mpl_toolkits.axes_grid1.ImageGrid(
                     fig, 111, # as in plt.subplot(111)
                     nrows_ncols=(self.nrows,self.ncols),
                     axes_pad=axes_pad,
                     share_all=True,
                     cbar_location="right",
                     cbar_mode="single",
                     cbar_size=cbar_size,
                     aspect=False)

        normalizer = matplotlib.colors.Normalize(*self._img_values_range)
        im=matplotlib.cm.ScalarMappable(norm=normalizer, cmap=self._cmap)

        plt_id = 0
        for i, row in enumerate(self._rows):
            for j in range(self.ncols):
                ax = grid[plt_id]
                plt_id += 1

                ax.imshow(row['imgs'][j], cmap=self._cmap, norm=normalizer, aspect='auto')
                ax.set_xticks([], [])
                ax.set_yticks([], [])
                if j==0 and row['y_title']:
                    kwds = {} if self._y_title_vertical else {'rotation': 0, 'labelpad': 40}
                    ax.set_ylabel(row['y_title'], **kwds)
                if row['x_titles'] and row['x_titles'][j]:
                    ax.set_title(row['x_titles'][j])
        cbar = ax.cax.colorbar(im)
        cbar.set_label(" ", labelpad=10) # This is how we get much needed padding on the right side of the figure.
        plt.figure(fig.number)
        plt.show() # this works better than fig.show() because it dumps immediately for %matplotlib inline
        #fig.show()
        if fig_path:
            print('Saving GridFigure to:', fig_path)
            fig.savefig(fig_path)

    @classmethod # for 1-row use case & backwards compatibility
    def display_3d(cls, *args, img_scale=default_img_scale, cmap=None, **kwd_args):
        row_figure = cls(img_scale=img_scale, cmap=cmap)
        row_figure.add_3d_row(*args, **kwd_args)
        row_figure.show()

    @classmethod # for 1-row use case & backwards compatibility
    def compare_img_seq(cls, *args, img_scale=default_img_scale, cmap=None, **kwd_args):
        row_figure = cls(img_scale=img_scale, cmap=cmap)
        row_figure.add_img_seq_row(*args, **kwd_args)
        row_figure.show()

    @classmethod # Verified to work: 5/10/24
    def display3d_dataset(cls, data_loader, n_sample_data=10, img_scale=default_img_scale, cmap=None):
        batch = next(iter(data_loader))
        print('batch["x"].shape=', batch['x'].shape)
        print('batch["y"].shape=', batch['y'].shape, '\n')

        import random
        for i, datum in enumerate(random.choices(data_loader.dataset, k=n_sample_data)):
            fig = cls(f"Sample datum i={i}:", img_scale=img_scale, cmap=cmap)
            datum['x'] = torch.as_tensor(datum['x']) # for permute
            datum['y'] = torch.as_tensor(datum['y']) # for permute
            x = datum['x'].permute([3,1,2,0])[0,:,:,:-3] # NOTE: we skip the 3 positional embedding channels
            fig.add_3d_row(x, y_title='input')
            fig.add_3d_row(datum['y'][0], y_title='output', x_title_func=lambda t: f't={t+x.shape[-1]}')
            fig.show()