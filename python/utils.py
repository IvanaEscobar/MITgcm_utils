import numpy as np
import matplotlib.pyplot as plt

def chunker(seq, size):
    """Divide 2D array into chunks -- used by get_blanklist"""
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def get_blanklist(landmask_faces, sNx, sNy, plot=False):
    """
    Extracts and returns the indices of blank tiles from a set of landmask faces.

    Parameters:
    - landmask_faces (dict): A dictionary containing landmask arrays for each face.
    - sNx (int): Size of tiles along the x-axis.
    - sNy (int): Size of tiles along the y-axis.
    - plot (bool, optional): If True, generates a visual representation of the blank tiles
    
    Returns:
    - blanklist (list): List of indices corresponding to blank tiles.

    Note:
    - The function also supports an optional plotting feature to visualize the tiles and their indices.

    Example:
    landmask_faces = {1: np.array(...), 2: np.array(...), ...}
    sNx = 32
    sNy = 32
    blanklist = get_blanklist(landmask_faces, sNx, sNy, plot=True)
    ```
    """
    
    # initialize plot
    if plot:
        text_kwargs = dict(ha='center', va='center', fontsize=10, color='r')
        fig, axes = plt.subplots(5,1)
    
    # initialize vars
    blanklist=[]
    tile_count = 0
    
    # loop through 5 faces. Note face_index = 1..5
    for face_index, landmask_face in landmask_faces.items():
        
        # create nan mask for plotting
        blanksmask_face = np.nan * np.ones_like(landmask_face)
        nx,ny = landmask_face.shape
        
        # start tile_count from total of prev face
        tile_count0 = tile_count
        
        # chunk face into tiles of size (sNx, sNy)
        for i, ii in enumerate(chunker(np.arange(nx),sNx)):
            for j, jj in enumerate(chunker(np.arange(ny),sNy)):
                    tile_count += 1                
                    # get this tile, check if all land
                    tile=landmask_face[np.ix_(ii, jj)]
                    isblank = tile.sum() == 0
                    
                    if isblank:
                        tile_index = tile_count0 + j+i*int(ny/sNy)+1
                        blanklist.append(tile_index)
                        blanksmask_face[np.ix_(ii, jj)]=0

                    # plot tile number text
                    if plot:
                        ax = axes.ravel()[face_index-1]
                        ax.text(jj[int(sNx/2)], ii[int(sNy/2)], '{}'.format(tile_count), **text_kwargs)
        
        # plot landmask, blanks
        if plot:
            aa=ax.contourf(landmask_face, cmap='Greys_r')
            aa=ax.pcolor(blanksmask_face, cmap='jet')
            
            # set ticks
            major_ticks_x = np.arange(0, ny, sNy)
            minor_ticks_x = np.arange(0, ny) 
            major_ticks_y = np.arange(0, nx, sNx)
            minor_ticks_y = np.arange(0, nx) 

            ax.set_xticks(major_ticks_x )
            ax.set_xticks(minor_ticks_x , minor=True)
            ax.set_yticks(major_ticks_y)
            ax.set_yticks(minor_ticks_y, minor=True)
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            
            ax.grid(which='minor', alpha=0.2)
            ax.grid(which='major', alpha=1)

            ax.set_title("Face {}".format(face_index))
            fig.set_size_inches(10,20)
    return blanklist
