# Picnic library script 'matplotlib_style.py'
# Author: Roger Hatfull
# Written at Los Alamos National Laboratories
# PCSRI Summer Program

def init_plot(style=None,nrows=1,ncols=1,columnwidth=241.14749,textheight=626.0,textwidth=506.295,frac=(1.,1.),dpi=100,wspace=0,hspace=0,fontsize=9.,**figargs):
    print("Initializing the plot...")
    #columnwidth = \the\columnwidth
    #textheight = \the\textheight
    #textwidth = \the\textwidth
    #frac = (fraction of columnwidth, fraction of textheight)

    import matplotlib as mpl

    columnwidth=columnwidth/72.*frac[0]
    textheight=textheight/72.*frac[1]
    textwidth=textwidth/72.*frac[0]

    m = mpl.rcParams
    if style == "onecolumn": # One column paper format
        # Specifically tailored for ACM Conference Proceedings LaTeX format,
        # but should work for any two-column LaTeX paper format.
        #m['font.size'] = 7.0
        if columnwidth * nrows > textheight:
            m['figure.figsize'] = (columnwidth,textheight)
        else:
            m['figure.figsize'] = (columnwidth,columnwidth * nrows)
        fontsize = 9
        dpi=150
        m['savefig.dpi'] = 300 # Need high enough resolution png images
    elif style == "twocolumn":
        m['figure.figsize'] = (textwidth,textheight)
        fontsize = 9
        dpi=150
        m['savefig.dpi'] = 300 # Need high enough resolution png images
    elif style == "poster":
        m['figure.figsize'] = (10.,10.)
        dpi = 100
        fontsize = 18
        m['savefig.dpi'] = 300 # Need high enough resolution png images

    m['axes.xmargin'] = 0.1
    m['axes.ymargin'] = m['axes.xmargin']

    m['font.size'] = fontsize
    m['figure.dpi'] = dpi
    # No space between subplots
    m['figure.subplot.wspace'] = wspace
    m['figure.subplot.hspace'] = hspace

    # Smaller scatter plot markers
    m['scatter.marker'] = '.'

    m['font.serif'] = 'Times New Roman'
    m['mathtext.fontset'] = 'stix'
    m['axes.titlesize'] = m['font.size']
    
    m['xtick.top'] = True
    m['xtick.bottom'] = True
    m['xtick.minor.visible'] = True
    m['ytick.right'] = True
    m['ytick.left'] = True
    m['ytick.minor.visible'] = True
    
    # Tick lengths scale with the font size to make it look pretty
    m['xtick.major.size'] = m['font.size']
    m['xtick.minor.size'] = m['xtick.major.size']*0.5
    m['xtick.direction'] = 'in'
    m['ytick.major.size'] = m['xtick.major.size']
    m['ytick.minor.size'] = m['xtick.minor.size']
    m['ytick.direction'] = m['xtick.direction']
    
    m['legend.fancybox'] = False

    #fig,ax = mpl.pyplot.subplots(nrows=nrows,ncols=ncols,**figargs)

    return mpl.pyplot.subplots(nrows=nrows,ncols=ncols,**figargs)
    



def get_visible_xticklabels(ax):
    import numpy as np
    # Here we want to consider only the ticklabels that are visible on the plot.
    # To do so, we need to omit any tick labels that apply to ticks which are
    # outside the plotted region.
    xticklabels = ax.get_xticklabels()
    if not xticklabels: return xticklabels
    axis_to_data = ax.transAxes + ax.transData.inverted()
    data_to_axis = axis_to_data.inverted()

    majortick_xlocs = np.array([ data_to_axis.transform([i,0.])[0] for i in ax.get_xticks() ])
    minpos = 0
    maxpos = len(majortick_xlocs)
    les = np.where(majortick_xlocs <= 0)[0]
    gtr = np.where(majortick_xlocs > 1)[0]
    if len(les) > 0:
        minpos = les[-1]+1
    if len(gtr) > 0:
        maxpos = gtr[0]

    majortick_xlocs = majortick_xlocs[minpos:maxpos]
    majortick_xlocs = np.round(majortick_xlocs,decimals=2)
        
    les = np.where(majortick_xlocs <= 0)[0]
    gtr = np.where(majortick_xlocs > 1)[0]
    if len(les)>0:
        minpos = les[-1]+1
    if len(gtr)>0:
        maxpos = gtr[0]

    #[ i.set_backgroundcolor("red") for i in xticklabels[minpos:maxpos] ]
    return xticklabels[minpos:maxpos]


def get_visible_yticklabels(ax):
    import numpy as np
    yticklabels = ax.get_yticklabels()
    if not yticklabels: return yticklabels
    axis_to_data = ax.transAxes + ax.transData.inverted()
    data_to_axis = axis_to_data.inverted()
    
    majortick_ylocs = np.array([ data_to_axis.transform([0.,i])[1] for i in ax.get_yticks() ])
    minpos = 0
    maxpos = len(majortick_ylocs)

    les = np.where(majortick_ylocs <= 0)[0]#+1
    gtr = np.where(majortick_ylocs > 1)[0]
    if len(les) > 0:
        minpos = les[-1]+1
    if len(gtr) > 0:
        maxpos = gtr[0]

    majortick_ylocs = majortick_ylocs[minpos:maxpos]
    majortick_ylocs = np.round(majortick_ylocs,decimals=2)

    les = np.where(majortick_ylocs <= 0)[0]
    gtr = np.where(majortick_ylocs > 1)[0]
    if len(les) > 0:
        minpos = les[-1]+1
    if len(gtr) > 0:
        maxpos = gtr[0]

    #[ i.set_backgroundcolor('red') for i in yticklabels[minpos:maxpos] ]
    return yticklabels[minpos:maxpos]


def get_axx0(axes,xwinbuffer0,ylabelbuffer):
    import numpy as np
    # Need to calculate the maximum label sizes to determine x0 and y0 axes positions.
    if type(axes).__module__ == np.__name__:
        axflatten = axes.flatten()
    else:
        axflatten = [axes]

    max_ylabelxsize = 0.
    for a in axflatten:
        fig = a.get_figure()
        renderer = fig.canvas.get_renderer()
        trans = fig.transFigure.inverted()
        for i in get_visible_yticklabels(a):
            labelpos = trans.transform(i.get_window_extent(renderer=renderer))
            max_ylabelxsize = max(max_ylabelxsize,labelpos[1][0]-labelpos[0][0])


    # We want the space between the axis labels and ticklabels to
    # be exact. So we will reshape the axis to make everything flush.
    # This will depend on the font size.
    ylabelfontsize = 0.
    for a in axflatten:
        ylabelfontsize = max(ylabelfontsize,a.yaxis.label.get_fontsize())
    fsx = ylabelfontsize/72. / fig.get_size_inches()[0]
    return xwinbuffer0 + fsx + max_ylabelxsize + ylabelbuffer

def get_axy0(axes,ywinbuffer0,xlabelbuffer):
    axflatten = axes.flatten()

    max_xlabelysize = 0.
    for a in axflatten:
        fig = a.get_figure()
        renderer = fig.canvas.get_renderer()
        trans = fig.transFigure.inverted()
        for i in get_visible_xticklabels(a):
            labelpos = trans.transform(i.get_window_extent(renderer=renderer))
            max_xlabelysize = max(max_xlabelysize,labelpos[1][1]-labelpos[0][1])


    # We want the space between the axis labels and ticklabels to
    # be exact. So we will reshape the axis to make everything flush.
    # This will depend on the font size.
    xlabelfontsize = 0.
    for a in axflatten:
        xlabelfontsize = max(xlabelfontsize,a.xaxis.label.get_size())
    fsy = xlabelfontsize/72. / fig.get_size_inches()[1]
    return ywinbuffer0 + fsy + max_xlabelysize + xlabelbuffer



def draw_axes(fig,ax,nrows,ncols,ndarray,axx0,axy0,xmax,ymax,wspace,hspace):
    import numpy as np
    if not isinstance(axx0,list):
        axx0 = np.zeros(ncols) + axx0

    width = (xmax - axx0[0] + wspace)/ncols - wspace
    height = (ymax - axy0 + hspace)/nrows - hspace
    if ndarray:
        if nrows > 1 and ncols == 1:
            for i in range(0,nrows):
                ax[nrows-i-1].set_position([axx0[0],axy0+i*(height+hspace),width,height])
        elif nrows == 1 and ncols > 1:
            for i in range(0,ncols):
                ax[i].set_position([axx0[i]+i*(width + wspace),axy0,width,height])
        else:
            for i in range(0,nrows):
                for j in range(0,ncols):
                    ax[nrows-i-1][j].set_position([axx0[j]+j*(width + wspace),axy0+i*(height+hspace),width,height])
    else:
        ax = ax[0]
        ax.set_position([axx0,axy0,width,height])
    fig.canvas.draw() # Update the axis locations
    



def fix_offsets(axes,figxsize):
    import matplotlib as mpl
    import numpy as np
    # If ticklabels are overlapping on the x-axis, we will set its formatter offset.
    # Then, the offset will be conveniently placed down in the axis label.
    # We will make sure the labels are separated by at least some minimum value.
    # The minimum value will be half the fontsize of the largest ticklabel.
    #figxsize = mpl.rcParams['figure.figsize'][0]

    #def format_func(value,tick_number):
    #    return '%1.0f' % (value*offset)

    
    axflatten = axes.flatten()
    """
    for a in axflatten:
        fig = a.get_figure()
        figxsize = fig.get_size_inches()[0]
        renderer = fig.canvas.get_renderer()
        trans = fig.transFigure.inverted()

        xticklabels = get_visible_xticklabels(a)
        if not xticklabels:
            continue

        minimum = max([ i.get_fontsize()/72./figxsize for i in xticklabels ])/2.

        overlap = False
        for i in range(1,len(xticklabels)-1):
            pos0 = trans.transform(xticklabels[i-1].get_window_extent(renderer=renderer))
            pos1 = trans.transform(xticklabels[i].get_window_extent(renderer=renderer))
            pos2 = trans.transform(xticklabels[i+1].get_window_extent(renderer=renderer))
            
            if (abs(pos0[1][0]-pos1[0][0]) < minimum or # Overlap on the left side of label i
                abs(pos1[1][0]-pos2[0][0]) < minimum): # Overlap on the right side of label i
                overlap = True

        labelvals = []
        for i in xticklabels:
            string = list(i.get_text())
            for c in range(0,len(string)):
                if ord(string[c]) == 8722: # Unicode character for minus sign
                    string[c] = "-"
            labelvals.append(float(''.join(string)))
        maxval = max([ i for i in labelvals ])

        offset = 1.

        formatter = a.xaxis.get_major_formatter()


        while overlap:
            print("Overlapping")
            if maxval < 0:
                offset = offset * 10
            else:
                offset = offset / 10

            
            #for i in xticklabels:
            #    string = list(i.get_text())
            #    for c in range(0,len(string)):
            #        if ord(string[c]) == 8722: # Unicode character for minus sign
            #            string[c] = "-"
            #
            #    string = str(float(''.join(string))*offset)
            #
            #    if len(string) > 16: # Avoid artificial precision
            #        newstring = list(string)
            #        if string.find("e") > -1:
            #            newstring[16:string.find("e")] = ""
            #        else:
            #            newstring[16:] = ""
            #        string = ''.join(newstring)
            #
            #    if float(string) == 0: string = str(abs(float(string))) # Remove silly negative zeros
            #
            #    if string.find("e") > -1: # Remove unnecessary zeros
            #        newstring = list(string)
            #        newstring[:string.find("e")] = list(''.join(newstring[:string.find("e")]).rstrip("0"))
            #        string = ''.join(newstring)
            #    else:
            #        string = string.rstrip("0")
            #
            #    # Remove trailing decimals if they're not necessary
            #    if list(string)[-1] == ".": string = ''.join(list(string)[:-1])
            #    i.set_text(string)
            #
            #a.set_xticklabels(xticklabels)
   
            
            #formatter.orderOfMagnitude = np.log10(1./offset)
            #print(formatter.pprint_val(100.))
            #a.xaxis.set_major_formatter(formatter)
            #print(formatter.orderOfMagnitude)

            #print([i.get_text() for i in xticklabels], offset)
            def format_func(value,tick_number):
                return '%1.0f' % (value*offset)

            a.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(format_func))
            print(a.xaxis.get_major_formatter().pprint_val(1000.))
            fig.canvas.draw()
            print([i.get_text() for i in xticklabels])
            xticklabels = get_visible_xticklabels(a)

            # Check again for overlap
            overlap = False
            for i in range(1,len(xticklabels)-1):
                pos0 = trans.transform(xticklabels[i-1].get_window_extent(renderer=renderer))
                pos1 = trans.transform(xticklabels[i].get_window_extent(renderer=renderer))
                pos2 = trans.transform(xticklabels[i+1].get_window_extent(renderer=renderer))

                if (abs(pos0[1][0]-pos1[0][0]) < minimum or # Overlap on the left side of label i
                    abs(pos1[1][0]-pos2[0][0]) < minimum): # Overlap on the right side of label i
                    overlap = True

    for a in axflatten:
        # If the data is being offset automatically by matplotlib, put that offset in
        # the axis labels instead so that it is visible.
        xtickoffset = a.get_xaxis().get_major_formatter().get_offset()
        ytickoffset = a.get_yaxis().get_major_formatter().get_offset()
        offsettext = "1"
        if xtickoffset.strip().replace(" ","") and a.get_xlabel():
            offsettext = xtickoffset
            a.xaxis.get_offset_text().set_visible(False)
        offsettext = str(float(offsettext)*offset)
        if offsettext and float(offsettext) != float(0.) and float(offsettext) != float(1.):
            a.set_xlabel(a.get_xlabel()+" ($\\times10^{" +str(int(np.log10(1./float(offsettext))))+"})$")

        
        if ytickoffset.strip().replace(" ","") and a.get_ylabel():
            ylabel = a.yaxis.get_label().get_text() + " ("+str(ytickoffset)+")"
            a.yaxis.get_offset_text().set_visible(False)
            a.set_ylabel(ylabel)
    """


    for a in axflatten:
        xtickoffset = a.get_xaxis().get_major_formatter().get_offset()
        ytickoffset = a.get_yaxis().get_major_formatter().get_offset()
        if xtickoffset.strip().replace(" ","") and a.get_xlabel():
            xlabel = a.xaxis.get_label().get_text() + " ("+str(xtickoffset)+")"
            a.xaxis.get_offset_text().set_visible(False)
            a.set_xlabel(xlabel)
        if ytickoffset.strip().replace(" ","") and a.get_ylabel():
            ylabel = a.yaxis.get_label().get_text() + " ("+str(ytickoffset)+")"
            a.yaxis.get_offset_text().set_visible(False)
            a.set_ylabel(ylabel)

            


def layout_legend(axes,minfontsize=None,framethreshold=0.3):
    # Modify the legend fontsize in case the legend is too long
    #
    # 'minfontsize' determines the minimum fontsize the legend text is allowed to have.
    # 'framethreshold' determines the maximum size the legend frame is allowed to have.
    #
    # This function will attempt to reduce the size of the legend frame by iteratively
    # reducing the fontsize of the label text until the frame is within the threshold.
    # If the fontsize reaches the minfontsize, no further reductions will happen and the
    # function will return.
    import numpy as np
    import matplotlib as mpl

    if minfontsize is None:
        minfontsize = mpl.rcParams['font.size']/2.

    if isinstance(axes,(np.ndarray,np.generic)):
        axflatten = axes.flatten()
    else:
        axflatten = [axes]
    for a in axflatten:
        fig = a.get_figure()
        trans = fig.transFigure.inverted()
        legend = a.get_legend()
        if legend is not None:
            if len(legend.get_texts()) != 0:
                legend_texts = legend.get_texts()
                fontsize = legend_texts[0].get_fontsize()

                frame = legend.get_frame()

                x0,y0 = trans.transform((frame.get_x(),frame.get_y()))
                w,h = trans.transform((frame.get_width(),frame.get_height()))
            
                axpos = a.get_position()
                while (w*h > axpos.width*axpos.height*framethreshold) and fontsize > minfontsize:
                    frame = legend.get_frame()
                    w,h = trans.transform((frame.get_width(),frame.get_height()))
                    fontsize -= min(1.,fontsize-minfontsize)
                    for i in legend_texts:
                        i.set_fontsize(fontsize)
                
                fig.canvas.draw()



#def get_all_children(fig):
#    children = fig.get_children()
#    all_children = []
#
#    def seek_all(children):
#        for child in children:
#            if isinstance(child,list):
#                seek_all(child.get_children())
#            else:
#                nextchildren = child.get_children()
#                if nextchildren:
#                    seek_all(nextchildren)
#                else:
#                    all_children.append(child)
#
#    seek_all(children)
#    return all_children



            
        



def auto_layout(fig,ax,nrows=1,ncols=1,xwinbuffer=[0.01,0.01],ywinbuffer=[0.01,0.01],xlabelbuffer=0.01,ylabelbuffer=0.025):
    print("Performing auto_layout...")
    # BUG: For some reason, doing the heatmap plots causes this function
    # to hang up severely.
    # xwinbuffer = (left, right)
    # ywinbuffer = (top, bottom)
    # If you supply 1 value for xwinbuffer or ywinbuffer, that value will
    # be used for both sides.

    import matplotlib as mpl
    import numpy as np
    from matplotlib.ticker import ScalarFormatter

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    trans = fig.transFigure.inverted()

    # In matplotlib, the wspace and hspace are taken to be the average width and height of the axes.
    # In this code, wspace and hspace are taken as figure coordinate quantities.
    # Thus, for example, wspace=0.5 will give a spacing between axes equal to half the size of the figure.
    wspace = mpl.rcParams['figure.subplot.wspace']
    hspace = mpl.rcParams['figure.subplot.hspace']

    fontsize = mpl.rcParams['font.size']
    figsize = mpl.rcParams['figure.figsize']

    if isinstance(xwinbuffer, float) or isinstance(xwinbuffer, int):
        xwinbuffer = [xwinbuffer,xwinbuffer]
    if isinstance(ywinbuffer, float) or isinstance(ywinbuffer, int):
        ywinbuffer = [ywinbuffer,ywinbuffer]

    if not hasattr(ax, "__len__"):
        axflatten = [ax]
    else:
        axflatten = ax.flatten()

    #for i in range(0,len(xwinbuffer)):
    #    xwinbuffer[i] = xwinbuffer[i]/ncols
    for i in range(0,len(ywinbuffer)):
        ywinbuffer[i] = ywinbuffer[i]/nrows


    maxsizex = -1.e30
    maxx = -1.e30
    extendingx = False
    maxsizey = -1.e30
    maxy=-1.e30
    extendingy = False

    ndarray = True

    if np.size(ax) == 1:
        ax = np.array([ax])
        ndarray = False




    if ncols > 1:
        axx0 = []
        if nrows == 1:
            for i in range(0,ncols):
                axx0.append(get_axx0(ax[i],xwinbuffer[0],ylabelbuffer))
        else:
            for i in range(0,ncols):
                axx0.append(get_axx0(ax[:,i],xwinbuffer[0],ylabelbuffer))
    else:
        axx0 = get_axx0(ax,xwinbuffer[0],ylabelbuffer)

    axy0 = get_axy0(ax,ywinbuffer[0],xlabelbuffer)



    # Set the new axis size
    xmax = 1. - xwinbuffer[1]
    ymax = 1. - ywinbuffer[1]

    draw_axes(fig,ax,nrows,ncols,ndarray,axx0,axy0,xmax,ymax,wspace,hspace)


    # Here, we make sure no xticklabels are overlapping by changing the offset on
    # the xaxis
    fix_offsets(ax,figsize[0])

    # Now that we have repositioned all the axes, we need to check to see if any axis ticklabels
    # are extending over. If so, we will adjust the axis positions to make the extending ticklabel
    # define xmax and ymax.
    max_over_x = 0.
    max_over_y = 0.
    max_xticklabel_x1 = 0.
    max_yticklabel_y1 = 0.
    for a in axflatten:
        if a.is_last_col():
            xticklabels = get_visible_xticklabels(a)
            pos = a.get_position()
            maxtemp = 0.
            for i in xticklabels:
                if i.get_visible():
                    labelpos = trans.transform(i.get_window_extent(renderer=renderer))
                    maxtemp = max(maxtemp,labelpos[1][0])
            max_xticklabel_x1 = max(max_xticklabel_x1,maxtemp)

        if a.is_first_row() or (nrows == 1 and a.is_last_row()):
            yticklabels = get_visible_yticklabels(a)
            pos = a.get_position()
            maxtemp = 0.
            for i in yticklabels:
                if i.get_visible():
                    labelpos = trans.transform(i.get_window_extent(renderer=renderer))
                    maxtemp = max(maxtemp,labelpos[1][1])
            max_yticklabel_y1 = max(max_yticklabel_y1,maxtemp)
    if max_xticklabel_x1 > xmax:
        xmax = xmax - (abs(xmax - max_xticklabel_x1))/ncols
    if max_yticklabel_y1 > ymax:
        ymax = ymax - (abs(ymax - max_yticklabel_y1))/nrows

    if ncols > 1:
        axx0 = []
        for i in range(0,ncols):
            axx0.append(get_axx0(ax[:][i],xwinbuffer[0],ylabelbuffer))
    else:
        axx0 = get_axx0(ax,xwinbuffer[0],ylabelbuffer)
    #axx0 = get_axx0(ax,xwinbuffer[0],ylabelbuffer)
    axy0 = get_axy0(ax,ywinbuffer[0],xlabelbuffer)



    # Redraw the axes
    draw_axes(fig,ax,nrows,ncols,ndarray,axx0,axy0,xmax,ymax,wspace,hspace)


    #mpl.pyplot.show()


    # We need to place the axis labels a fixed distance away from the ticklabels.
    # That distance is, for ylabels, ax.get_position().x0 - max_ylabelxsize - fsx*xlabelbuffer
    # for xlabels, it is ax.get_position().y0 - max_xlabelysize - fsy*ylabelbuffer
    if ncols > 1 and nrows > 1:
        for j in range(0,ncols):
            maxtemp2 = 0.

            for i in range(0,nrows):
                maxtemp = 0.
                for k in get_visible_yticklabels(ax[i][j]):
                    labelpos = trans.transform(k.get_window_extent(renderer=renderer))
                    maxtemp = max(maxtemp,labelpos[1][0]-labelpos[0][0])
                maxtemp2 = max(maxtemp2,maxtemp)
            max_ylabelxsize = maxtemp2

            for i in range(0,nrows):
                ylabel = ax[i][j].yaxis.get_label().get_text()
                if ylabel != "":
                    pos = ax[i][j].get_position()
                    ylabelpos = pos.x0 - ylabelbuffer - max_ylabelxsize
                    if not ax[i][j].is_first_col():
                        backpos = ax[i][j-1].get_position()
                        if ylabelpos - ax[i][j].yaxis.get_label().get_fontsize()/72./figsize[0] <= backpos.x1:
                            ylabelpos = (backpos.x1 + ax[i][j].yaxis.get_label().get_fontsize()/72./figsize[0] + pos.x0 - max_ylabelxsize)/2.
                    else:
                        ylabelpos = xwinbuffer[0] + ax[i][j].yaxis.get_label().get_fontsize()/72./figsize[0]

                    figure_to_axis = ax[i][j].get_figure().transFigure + ax[i][j].transAxes.inverted()
                    ylabelpos = figure_to_axis.transform((ylabelpos,(pos.y1+pos.y0)/2.))
                    ax[i][j].yaxis.set_label_coords(ylabelpos[0],ylabelpos[1])
    elif ncols > 1 and nrows == 1:
        for j in range(0,ncols):
            maxtemp2 = 0.
            maxtemp = 0.
            for k in get_visible_yticklabels(ax[j]):
                labelpos = trans.transform(k.get_window_extent(renderer=renderer))
                maxtemp = max(maxtemp,labelpos[1][0]-labelpos[0][0])
            maxtemp2 = max(maxtemp2,maxtemp)
            max_ylabelxsize = maxtemp2

            ylabel = ax[j].yaxis.get_label().get_text()
            if ylabel != "":
                pos = ax[j].get_position()
                ylabelpos = pos.x0 - ylabelbuffer - max_ylabelxsize
                if not ax[j].is_first_col():
                    backpos = ax[j-1].get_position()
                    if ylabelpos - ax[j].yaxis.get_label().get_fontsize()/72./figsize[0] <= backpos.x1:
                        ylabelpos = (backpos.x1 + ax[j].yaxis.get_label().get_fontsize()/72./figsize[0] + pos.x0 - max_ylabelxsize)/2.
                else:
                    ylabelpos = xwinbuffer[0] + ax[j].yaxis.get_label().get_fontsize()/72./figsize[0]

                figure_to_axis = ax[j].get_figure().transFigure + ax[j].transAxes.inverted()
                ylabelpos = figure_to_axis.transform((ylabelpos,(pos.y1+pos.y0)/2.))
                ax[j].yaxis.set_label_coords(ylabelpos[0],ylabelpos[1])

    else:
        max_ylabelxsize = 0.
        for a in axflatten:
            for i in get_visible_yticklabels(a):
                labelpos = trans.transform(i.get_window_extent(renderer=renderer))
                max_ylabelxsize = max(max_ylabelxsize,labelpos[1][0]-labelpos[0][0])
        for a in axflatten:
            ylabel = a.yaxis.get_label().get_text()
            if ylabel != "":
                pos = a.get_position()
                ylabelpos_axunit = - (ylabelbuffer+max_ylabelxsize) / pos.width
                a.yaxis.set_label_coords(ylabelpos_axunit,0.5)


    if nrows > 1:
        for j in range(0,nrows):
            maxtemp2 = 0.
            if ncols > 1:
                axuse = ax[j]
                if j < nrows-1:
                    backaxuse = ax[j+1]
                else:
                    backaxuse = None
            else:
                axuse = [ax[j]]
                if j < nrows-1:
                    backaxuse = [ax[j+1]]
                else:
                    backaxuse = None

            for i in range(0,ncols):
                maxtemp = 0.
                for k in get_visible_xticklabels(axuse[i]):
                    labelpos = trans.transform(k.get_window_extent(renderer=renderer))
                    maxtemp = max(maxtemp,labelpos[1][1]-labelpos[0][1])
                maxtemp2 = max(maxtemp2,maxtemp)
            max_xlabelysize = maxtemp2

            for i in range(0,ncols):
                xlabel = axuse[i].xaxis.get_label().get_text()
                if xlabel != "":
                    pos = axuse[i].get_position()
                    xlabelpos = pos.y0 - xlabelbuffer - max_xlabelysize
                    if not axuse[i].is_last_row():
                        backpos = backaxuse[i].get_position()
                        if xlabelpos - axuse[i].xaxis.get_label().get_fontsize()/72./figsize[1] <= backpos.y1:
                            xlabelpos = (backpos.y1 + axuse[i].xaxis.get_label().get_fontsize()/72./figsize[1] + pos.y0 - max_xlabelysize)/2.
                    else:
                        xlabelpos = ywinbuffer[0] + axuse[i].xaxis.get_label().get_fontsize()/72./figsize[1]

                    figure_to_axis = axuse[i].get_figure().transFigure + axuse[i].transAxes.inverted()
                    xlabelpos = figure_to_axis.transform(((pos.x1+pos.x0)/2.,xlabelpos))
                    axuse[i].xaxis.set_label_coords(xlabelpos[0],xlabelpos[1])
    else:
        max_xlabelysize = 0.
        for a in axflatten:
            for i in get_visible_xticklabels(a):
                labelpos = trans.transform(i.get_window_extent(renderer=renderer))
                max_xlabelysize = max(max_xlabelysize,labelpos[1][1]-labelpos[0][1])
        for a in axflatten:
            xlabel = a.xaxis.get_label().get_text()
            if xlabel != "":
                pos = a.get_position()
                xlabelpos_axunit = - (xlabelbuffer+max_xlabelysize) / pos.height
                a.xaxis.set_label_coords(0.5,xlabelpos_axunit)
            

    fig.canvas.draw()





    layout_legend(ax)
            
            
def autolabel(rects, xpos='center'):
    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0, 'right': 1, 'left': -1}
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(offset[xpos]*3, 3),  # use 3 points offset
                    textcoords="offset points",  # in both directions
                    ha=ha[xpos], va='bottom')

