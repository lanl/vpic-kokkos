pro doplot,outform
common choice, plotme
common parameters,nx,ny,tslices,xmax,ymax,twci,rhoi,teti,mime,wpewce,numq
common picdata, field, struct
common colortable,rgb,usecolor,red,blue,green,tmax,tmin,fooa1,fooa2,range1,range2,facmax,facmin,r1,r2,foob1,tmaxb,tminb,ra1,ra2,ri1,ri2
common controlplot,v
common imagesize,nxpix,nypix,xoff,yoff,xpic,ypic


; CHANGE NX NY L1032

; Temporary storage

fulldata = fltarr(nx,ny)

; Read needed data for contour plot

print
print,'--------------------------------'
print,"Output Record=",v.quantity
print,"   Time Index=",v.time
field = assoc(v.quantity,struct)
mytitle = plotme(v.quantity)
struct = field[v.time]
;simulation_time=struct.time/(wpewce*mime)
simulation_time=v.time*twci
print,"t*wci=",simulation_time
;print,"it=",struct.it
if ( outform eq 0 ) then print, "Output form = X11"
if ( outform eq 1 ) then print, "Output form = Postscript"
fulldata=struct.data

;  Substract initial equilibrium is desired

if ( v.data eq 1 and v.time gt 0 ) then begin
    print," *** Subtract initial value, to give perturbation ***"
    mytitle = '('+plotme(v.quantity)+'-'+plotme(v.quantity)+'[t=0])'
    struct = field[0]
    fulldata = fulldata - struct.data
endif

;  Multiply by another quantity if desired

if ( v.multiply gt 0 ) then begin
    print,"Mutiply data with = ",plotme(v.multiply)
    mytitle = mytitle+"*"+plotme(v.multiply)
    field = assoc(v.multiply,struct)
    struct = field[v.time]
    fulldata = fulldata*struct.data
endif

; Divide by another quantity if desired

if ( v.divide gt 0 ) then begin
    print,"Divide data with ",plotme(v.divide)
    mytitle = mytitle+"/"+plotme(v.divide)
    field = assoc(v.divide,struct)
    struct = field[v.time]
    for i=0,nx-1 do begin
        for j=0,ny-1 do begin
            if ( abs(struct.data(i,j)) gt 0.0 ) then fulldata(i,j) = fulldata(i,j)/struct.data(i,j)
        endfor
    endfor
endif

;  Now shift data the required amount in y direction

fulldata = shift(fulldata,0,v.shift)

; Select region of 2d-data to plot - This allows me to create contours just within the specified
; region and ignore the rest of the data

jmin=fix(v.ymin/ymax*ny)
jmax=fix(v.ymax/ymax*ny)-1
ly = jmax-jmin +1 
imin =fix(v.xmin/xmax*nx)
imax =fix(v.xmax/xmax*nx)-1
lx = imax-imin +1 

; Declare memory for reduced region and load region into plotting array

temp = fltarr(lx,ly)
temp = fulldata(imin:imax,jmin:jmax)

; Output info about data

print,'Maximum Value=',max(abs(temp))

; Smooth the temp
  
if (v.smoothing le 8) then begin
    print,"Smoothing=",v.smoothing
    temp = smooth(temp,v.smoothing)
endif

; Create x and y coordinates

xarr = v.ymin + ((v.ymax-v.ymin)/ly)*(findgen(ly)+0.5)
yarr = ((v.xmax-v.xmin)/lx)*(findgen(lx)+0.5) + v.xmin

; set output for either X11 or a postscript file

if ( outform eq 0 ) then begin
    set_plot, 'X'
    device,true_color=24,decomposed=0	
    !x.style=1
    !y.style=1
    !p.color =0
    !p.background=1
endif

if (outform eq 1 ) then begin
    set_plot, 'ps'
    !p.color =1
    !p.background = 0
    !p.font = 0	
    !x.style=1
    !y.style=1
    width=5.0
    asp=0.5
    height=width*asp
    if ( v.plottype eq 0 ) then file=strcompress(plotme(v.quantity)+string(v.time)+'.eps',/remove_all)
    if ( v.plottype eq 1 ) then file=strcompress(plotme(v.quantity)+string(v.time)+'_ave.eps',/remove_all)
    if ( v.plottype ge 2 ) then file=strcompress(plotme(v.quantity)+string(v.time)+'.eps',/remove_all)
    device, /encapsulated, bits_per_pixel=32, filename = file, $  
            /inches, xsize=width, ysize = height, /TIMES, font_size=12, /color, set_font='Times-Roman'
endif

;  Set colors so that red is largest value

ri1=-94.86
ri2=123.57

shady=transpose(temp)
if ( outform ne 2 ) then begin
    r1=min(shady)
    r2=max(shady)
    dr = (r2-r1)*0.32
;   r1=ri1
;   r2=ri2
;   dr=0
    r1=r1-dr/2
    r2=r2+dr/2
   ; r1 = -0.0005
   ; r2 = 0.002
    fooa1=min(temp)
    fooa2=max(temp)
    if (v.map eq 0) then begin
;        if (abs(fooa1) gt abs(fooa2)) then usecolor=reverse(rgb) else usecolor=rgb 
        usecolor=rgb 
        red=usecolor(0,*)
        green=usecolor(1,*)
        blue=usecolor(2,*)
        tvlct,red,green,blue
    endif
    tmax=max(temp-fooa1)*facmax
    tmin=facmin*tmax
endif

; Contour parameters

    print,'Contours=',v.contours
    temp = (temp - fooa1)
    tcolor=v.contours
    step=(tmax-tmin)/tcolor
    clevels=indgen(tcolor)*step + tmin

; Y averaged data at t=0 

yave0 = fltarr(lx)
struct = field[0]
temp0 = struct.data(imin:imax,jmin:jmax)
for i=0,lx-1 do begin
    yave0(i) = 0.0
    for j=0,ly-1 do begin
        yave0(i) = yave0(i) +  temp0(i,j)
    endfor
endfor
yave0 = yave0/ly

; Now create the desired plot

if ( v.plottype eq 0 ) then begin

; Set position and titles

    xoff = 0.10
    yoff = 0.15
    xpic = 0.87
    ypic = 0.87
    dx1=0.012
    dx2=0.045
    !x.title="z/L"
    !y.title="x/L"
    !p.title=mytitle+"      t*Wci="+strcompress(string(FORMAT='(f6.2)',simulation_time))+"     Index="+strcompress(string(v.time),/remove_all)
    !p.position=[xoff,yoff,xpic,ypic] 
;    contour, transpose(temp), xarr, yarr, charsize =  1.4,levels=clevels, /fill, color=1


    !p.title=""
    shade_surf, shady, xarr, yarr, ax=90,az=0,shades=bytscl(shady,max=r2,min=r1),zstyle=4,charsize=1.5
    xyouts,v.ymin+(v.ymax-v.ymin)/3.2,(v.xmax)*1.03,mytitle+"   t*Wci="+strcompress(string(FORMAT='(f8.2)',simulation_time))+"     Index="+strcompress(string(v.time),/remove_all),charsize=2.0


           
;  Now add the color bar

    temp = temp + fooa1
    !x.title=""
    !y.title=""
    !p.title=""
    colorbar, Position = [xpic+dx1,yoff,xpic+dx2,ypic], /Vertical, $
              Range=[r1,r2], Format='(f9.2)', /Right, $
              Bottom=5, ncolors=221, Divisions=6, font=9

endif

if ( v.plottype ge 2 ) then begin

; Set position and titles
    
    xoff = 0.07
    yoff = 0.11
    xpic = 0.60
    ypic = 0.93
    !p.position=[xoff,yoff,xpic,ypic]
    !x.title="y/L"
    !y.title="x/L"
    !p.title=mytitle+"      t*Wci="+strcompress(string(FORMAT='(f8.2)',simulation_time))+"     Index="+strcompress(string(v.time),/remove_all)
    !p.position=[xoff,yoff,xpic,ypic]     
;    contour, transpose(temp), xarr, yarr, charsize =
;    1.4,levels=clevels, /fill, color=1
    !p.title=""
    shade_surf, shady, xarr, yarr, ax=90,az=0,shades=bytscl(shady,max=r2,min=r1),zstyle=4,charsize=1.5
    xyouts,v.ymin+(v.ymax-v.ymin)/3.2,(v.xmax)*1.03,mytitle+"    t*Wci="+strcompress(string(FORMAT='(f8.2)',simulation_time))+"     Index="+strcompress(string(v.time),/remove_all),charsize=2.0


    temp = temp + fooa1
endif

; Overplot a line to show where the y-slice is located

if ( v.plottype eq 3) then begin
    xs=fltarr(lx)
    for i=0,lx-1 do begin
        xs(i) = v.yslice
    endfor
    oplot,xs,yarr
endif

; Overplot a line to show where the x-slice is located

if ( v.plottype eq 4) then begin
    xs=fltarr(ly)
    for i=0,ly-1 do begin
        xs(i) = v.xslice
    endfor
    oplot,xarr,xs
endif

;  Finally, overplot lines of another quantity if requested

if ( v.overplot gt 0 ) then begin
    print,"Overplot with = ",plotme(v.overplot)
    field = assoc(v.overplot,struct)
    struct = field[v.time]
    fulldata = shift(struct.data,0,v.shift)
    temp2 = fltarr(lx,ly)	
    temp2 = fulldata(imin:imax,jmin:jmax)
    if (v.smoothing le 8) then temp2 = smooth(temp2,v.smoothing)
    if (outform ne 2) then begin
        foob1 = min(temp2)
        tmaxb=max(temp2)
        tminb=min(temp2)
    endif
    step=(tmaxb-tminb)/(tcolor)
    clevels=indgen(tcolor)*step + tminb        
    contour, transpose(temp2), xarr, yarr,levels=clevels, /overplot,color=1
endif

; Y averaged data

yave = fltarr(lx)
if ( v.plottype eq 1 or v.plottype eq 2 ) then begin
    for i=0,lx-1 do begin
        yave(i) = 0.0
        for j=0,ly-1 do begin
            yave(i) = yave(i) +  temp(i,j)
        endfor
    endfor
endif
yave = yave/ly

; Y slice of data

if ( v.plottype eq 3) then begin
    jslice=fix((v.yslice-v.ymin)/(v.ymax-v.ymin)*(ly-1))
    for i=0,lx-1 do begin
        yave(i) = temp(i,jslice)
    endfor

    if ( v.overplot gt 0 ) then begin
        yave2 = fltarr(lx)
        for i=0,lx-1 do begin
            yave2(i) = temp2(i,jslice)
        endfor
    endif

endif

; X slice of data

if ( v.plottype eq 4) then begin
    yave = fltarr(ly)
    islice=fix((v.xslice-v.xmin)/(v.xmax-v.xmin)*(lx-1))
    for j=0,ly-1 do begin
        yave(j) = temp(islice,j)
    endfor

    if ( v.overplot gt 0 ) then begin
        yave2 = fltarr(ly)
        for j=0,ly-1 do begin
            yave2(j) = temp2(islice,j)
        endfor
    endif

endif

; Set range for line plot

if ( outform eq 0) then begin
    range1=min(yave)
    range2=max(yave)
    dr = (range2-range1)*0.20
    range1=range1-dr/2
    range2=range2+dr/2
;  range1=ra1
;  range2=ra2
endif

; Create y-average plot

if (v.plottype eq 1) then begin
    xoff2 = 0.13
    yoff2 = 0.13
    xpic2 = 0.92
    ypic2 = 0.92
    !x.title="x/L"
    !y.title=mytitle
    !p.title=mytitle+"      t*Wci="+strcompress(string(FORMAT='(f8.2)',simulation_time))+"     Index="+strcompress(string(v.time),/remove_all)
    !p.position=[xoff2,yoff2,xpic2,ypic2] 
    plot, yarr,yave,charsize=1.4,xrange=[v.xmin,v.xmax],yrange=[range1,range2]
endif

if (v.plottype eq 2 or v.plottype eq 3) then begin

    xoff2 = 0.69
    yoff2 = 0.11
    xpic2 = 0.98
    ypic2 = 0.93
    !y.title="x/L"
    if (v.plottype eq 2) then !x.title="<"+mytitle+">"
    if (v.plottype eq 3) then !x.title=strcompress(mytitle+"(y="+string(FORMAT='(f5.1)',v.yslice)+")",/remove_all)
    !p.title=mytitle+"      t*Wci="+strcompress(string(FORMAT='(f8.2)',simulation_time))
    !p.position=[xoff2,yoff2,xpic2,ypic2] 
    plot, yave,yarr,charsize=1.4,yrange=[v.xmin,v.xmax],xrange=[range1,range2],/NOERASE
;,xrange=[range1,range2],/NOERASE

;    oplot,yave0,yarr,color=215

    if ( v.overplot gt 0 ) then  oplot,yave2,yarr,color=55

endif

if (v.plottype eq 4) then begin

    xoff2 = 0.69
    yoff2 = 0.11
    xpic2 = 0.98
    ypic2 = 0.93
    !x.title="y/L"
    !y.title="<"+mytitle+">"
    !p.title=mytitle+"      t*Wci="+strcompress(string(FORMAT='(f8.2)',simulation_time))
    !p.position=[xoff2,yoff2,xpic2,ypic2] 
    plot, xarr,yave,charsize=1.4,xrange=[v.ymin,v.ymax],yrange=[range1,range2],/NOERASE
;,xrange=[range1,range2],/NOERASE

    if ( v.overplot gt 0 ) then  oplot,xarr,yave2,color=55

endif

end


pro handle_event, ev
common controlplot,v
common parameters,nx,ny,tslices,xmax,ymax,twci,rhoi,teti,mime,wpewce,numq
common imagesize,nxpix,nypix,xoff,yoff,xpic,ypic
common parameters,nx,ny,tslices,xmax,ymax,twci,rhoi,teti,mime,wpewce,numq
common oldposition,xmax0,ymin0
widget_control, ev.id, get_uval = whichbutton
case whichbutton of
    'done' : begin
        close,/All
        free_lun, 1
        widget_control, ev.top, /destroy
    end
    'plot' : begin
        widget_control,ev.top,get_uvalue=v
        v.xmin=0
        v.xmax=xmax
        v.ymin=0.0
        v.ymax=ymax
        widget_control,ev.top,set_uvalue=v    
        doplot,0
    end
    'render' : begin
        widget_control,ev.top,get_uvalue=v
        doplot,1
    end
    'animate' : begin
        widget_control,ev.top,get_uvalue=v
        create_movie
    end
    'region' : begin
        widget
    end

    'mouse' : begin
        widget_control,ev.top,get_uvalue=v
        if (ev.type eq 0) then begin
            xmax0=v.xmax
            ymin0=v.ymin
            v.xmax=(float(ev.y)/float(nypix)-yoff)*(v.xmax-v.xmin)/(ypic-yoff)+v.xmin
            v.ymin=(float(ev.x)/float(nxpix)-xoff)*(v.ymax-v.ymin)/(xpic-xoff)+v.ymin
;            if (v.xmax gt xmax) then v.xmax=xmax
            if (v.ymin lt 0.0) then v.ymin=0.
            widget_control,ev.top,set_uvalue=v    
            print,"xmax=",v.xmax," ymin=",v.ymin
        endif
        if (ev.type eq 1) then begin
            v.xmin=(float(ev.y)/float(nypix)-yoff)*(xmax0-v.xmin)/(ypic-yoff)+v.xmin
            v.ymax=(float(ev.x)/float(nxpix)-xoff)*(v.ymax-ymin0)/(xpic-xoff)+ymin0
            if (v.xmin lt -xmax/2.0) then v.xmin=-xmax/2.0
            if (v.ymax gt ymax) then v.ymax=ymax
            widget_control,ev.top,set_uvalue=v  
            print,"xmin=",v.xmin," ymax=",v.ymax  
            doplot,0
        endif
    end
    
    'quantity' : begin
        widget_control,ev.top,get_uvalue=v                	
        v.quantity=ev.index
        widget_control,ev.top,set_uvalue=v    
    end

    'multiply' : begin
        widget_control,ev.top,get_uvalue=v                	
        v.multiply=ev.index
        widget_control,ev.top,set_uvalue=v    
    end

    'divide' : begin
        widget_control,ev.top,get_uvalue=v                	
        v.divide=ev.index
        widget_control,ev.top,set_uvalue=v    
    end

    'overplot' : begin
        widget_control,ev.top,get_uvalue=v                	
        v.overplot=ev.index
        widget_control,ev.top,set_uvalue=v    
    end

    'options' : begin
        widget_control,ev.top,get_uvalue=v                	
        print," Selected Option = ",ev.value       
        if ( ev.value ge 2 ) and ( ev.value le 9 ) then v.smoothing=ev.value
        if ( ev.value ge 11 ) and ( ev.value le 30) then v.contours=(ev.value*2)-10
        if ( ev.value eq 31 ) then v.data=0 
        if ( ev.value eq 32 ) then v.data=1
        if ( ev.value ge 35 ) then v.map=35-ev.value
        if ( ev.value eq 36 ) then xloadct,bottom=5
        widget_control,ev.top,set_uvalue=v    
        
    end


    'ptype' : begin
        widget_control,ev.top,get_uvalue=v                	
        v.plottype=ev.index
        widget_control,ev.top,set_uvalue=v
        doplot,0
    end

    'time' : begin
        widget_control,ev.top,get_uvalue=v                	 	
        v.time=ev.value
        widget_control,ev.top,set_uvalue=v 
        doplot,0
    end

    'shift' : begin
        widget_control,ev.top,get_uvalue=v                	 	
        v.shift=ev.value
        widget_control,ev.top,set_uvalue=v 
        doplot,0
    end

    'ymin' : begin
        widget_control,ev.top,get_uvalue=v                	
        v.ymin=ev.value
        widget_control,ev.top,set_uvalue=v    
    end

    'ymax' : begin
        widget_control,ev.top,get_uvalue=v                	
        v.ymax=ev.value
        widget_control,ev.top,set_uvalue=v    
    end

    'yslice' : begin
        widget_control,ev.top,get_uvalue=v                	
        v.yslice=ev.value
        widget_control,ev.top,set_uvalue=v    
    end

    'xslice' : begin
        widget_control,ev.top,get_uvalue=v                	
        v.xslice=ev.value
        widget_control,ev.top,set_uvalue=v    
    end

    'xmax' : begin
        widget_control,ev.top,get_uvalue=v                	
        v.xmax=ev.value
        v.xmin=-ev.value
        widget_control,ev.top,set_uvalue=v    
    end

endcase

end



; NAME: COLORBAR
; PURPOSE:
;       The purpose of this routine is to add a color bar to the current
;       graphics window.
; AUTHOR:
;   FANNING SOFTWARE CONSULTING
;   David Fanning, Ph.D.
;   2642 Bradbury Court
;   Fort Collins, CO 80521 USA
;   Phone: 970-221-0438
;   E-mail: davidf@dfanning.com
;   Coyote's Guide to IDL Programming: http://www.dfanning.com/
; CATEGORY:
;       Graphics, Widgets.
; CALLING SEQUENCE:
;       COLORBAR
; INPUTS:
;       None.
; KEYWORD PARAMETERS:
;       BOTTOM:   The lowest color index of the colors to be loaded in
;                 the bar.
;       CHARSIZE: The character size of the color bar annotations. Default is 1.0.
;
;       COLOR:    The color index of the bar outline and characters. Default
;                 is !P.Color..
;       DIVISIONS: The number of divisions to divide the bar into. There will
;                 be (divisions + 1) annotations. The default is 6.
;       FONT:     Sets the font of the annotation. Hershey: -1, Hardware:0, 
;                 True-Type: 1.
;       FORMAT:   The format of the bar annotations. Default is '(I5)'.
;       MAXRANGE: The maximum data value for the bar annotation. Default is
;                 NCOLORS.
;       MINRANGE: The minimum data value for the bar annotation. Default is 0.
;       MINOR:    The number of minor tick divisions. Default is 2.
;       NCOLORS:  This is the number of colors in the color bar.
;
;       POSITION: A four-element array of normalized coordinates in the same
;                 form as the POSITION keyword on a plot. Default is
;                 [0.88, 0.15, 0.95, 0.95] for a vertical bar and
;                 [0.15, 0.88, 0.95, 0.95] for a horizontal bar.
;       RANGE:    A two-element vector of the form [min, max]. Provides an
;                 alternative way of setting the MINRANGE and MAXRANGE keywords.
;       RIGHT:    This puts the labels on the right-hand side of a vertical
;                 color bar. It applies only to vertical color bars.
;       TITLE:    This is title for the color bar. The default is to have
;                 no title.
;       TOP:      This puts the labels on top of the bar rather than under it.
;                 The keyword only applies if a horizontal color bar is rendered.
;       VERTICAL: Setting this keyword give a vertical color bar. The default
;                 is a horizontal color bar.
; COMMON BLOCKS:
;       None.
; SIDE EFFECTS:
;       Color bar is drawn in the current graphics window.
; RESTRICTIONS:
;       The number of colors available on the display device (not the
;       PostScript device) is used unless the NCOLORS keyword is used.
; EXAMPLE:
;       To display a horizontal color bar above a contour plot, type:
;       LOADCT, 5, NCOLORS=100
;       CONTOUR, DIST(31,41), POSITION=[0.15, 0.15, 0.95, 0.75], $
;          C_COLORS=INDGEN(25)*4, NLEVELS=25
;       COLORBAR, NCOLORS=100, POSITION=[0.15, 0.85, 0.95, 0.90]
; MODIFICATION HISTORY:
;       Written by: David Fanning, 10 JUNE 96.
;       10/27/96: Added the ability to send output to PostScript. DWF
;       11/4/96: Substantially rewritten to go to screen or PostScript
;           file without having to know much about the PostScript device
;           or even what the current graphics device is. DWF
;       1/27/97: Added the RIGHT and TOP keywords. Also modified the
;            way the TITLE keyword works. DWF
;       7/15/97: Fixed a problem some machines have with plots that have
;            no valid data range in them. DWF
;       12/5/98: Fixed a problem in how the colorbar image is created that
;            seemed to tickle a bug in some versions of IDL. DWF.
;       1/12/99: Fixed a problem caused by RSI fixing a bug in IDL 5.2. 
;                Sigh... DWF.
;       3/30/99: Modified a few of the defaults. DWF.
;       3/30/99: Used NORMAL rather than DEVICE coords for positioning bar. DWF.
;       3/30/99: Added the RANGE keyword. DWF.
;       3/30/99: Added FONT keyword. DWF
;       5/6/99: Many modifications to defaults. DWF.
;       5/6/99: Removed PSCOLOR keyword. DWF.
;       5/6/99: Improved error handling on position coordinates. DWF.
;       5/6/99. Added MINOR keyword. DWF.
;       5/6/99: Set Device, Decomposed=0 if necessary. DWF.
;       2/9/99: Fixed a problem caused by setting BOTTOM keyword, but not 
;               NCOLORS. DWF.
;       8/17/99. Fixed a problem with ambiguous MIN and MINOR keywords. DWF
;       8/25/99. I think I *finally* got the BOTTOM/NCOLORS thing sorted out. 
;                :-( DWF.
;       10/10/99. Modified the program so that current plot and map 
;                 coordinates are
;            saved and restored after the colorbar is drawn. DWF.
;       3/18/00. Moved a block of code to prevent a problem with color 
;                decomposition. DWF.
;       4/28/00. Made !P.Font default value for FONT keyword. DWF.
;###########################################################################
; LICENSE
;
; This software is OSI Certified Open Source Software.
; OSI Certified is a certification mark of the Open Source Initiative.
;
; Copyright © 2000 Fanning Software Consulting.
;
; This software is provided "as-is", without any express or
; implied warranty. In no event will the authors be held liable
; for any damages arising from the use of this software.
; Permission is granted to anyone to use this software for any
; purpose, including commercial applications, and to alter it and
; redistribute it freely, subject to the following restrictions:
; 1. The origin of this software must not be misrepresented; you must
;    not claim you wrote the original software. If you use this software
;    in a product, an acknowledgment in the product documentation
;    would be appreciated, but is not required.
; 2. Altered source versions must be plainly marked as such, and must
;    not be misrepresented as being the original software.
; 3. This notice may not be removed or altered from any source distribution.
; For more information on Open Source Software, visit the Open Source
; web site: http://www.opensource.org.
;###########################################################################
PRO COLORBAR, BOTTOM=bottom, CHARSIZE=charsize, COLOR=color, $
   DIVISIONS=divisions, $
   FORMAT=format, POSITION=position, MAXRANGE=maxrange, MINRANGE=minrange, $
   NCOLORS=ncolors, $
   TITLE=title, VERTICAL=vertical, TOP=top, RIGHT=right, MINOR=minor, $
   RANGE=range, FONT=font, TICKLEN=ticklen, _EXTRA=extra

   ; Return to caller on error.
On_Error, 2

   ; Save the current plot state.

bang_p = !P
bang_x = !X
bang_Y = !Y
bang_Z = !Z
bang_Map = !Map

   ; Is the PostScript device selected?

postScriptDevice = (!D.NAME EQ 'PS' OR !D.NAME EQ 'PRINTER')

   ; Which release of IDL is this?

thisRelease = Float(!Version.Release)

    ; Check and define keywords.

IF N_ELEMENTS(ncolors) EQ 0 THEN BEGIN

   ; Most display devices to not use the 256 colors available to
   ; the PostScript device. This presents a problem when writing
   ; general-purpose programs that can be output to the display or
   ; to the PostScript device. This problem is especially bothersome
   ; if you don't specify the number of colors you are using in the
   ; program. One way to work around this problem is to make the
   ; default number of colors the same for the display device and for
   ; the PostScript device. Then, the colors you see in PostScript are
   ; identical to the colors you see on your display. Here is one way to
   ; do it.

   IF postScriptDevice THEN BEGIN
      oldDevice = !D.NAME

         ; What kind of computer are we using? SET_PLOT to appropriate
         ; display device.

      thisOS = !VERSION.OS_FAMILY
      thisOS = STRMID(thisOS, 0, 3)
      thisOS = STRUPCASE(thisOS)
      CASE thisOS of
         'MAC': SET_PLOT, thisOS
         'WIN': SET_PLOT, thisOS
         ELSE: SET_PLOT, 'X'
      ENDCASE

         ; Here is how many colors we should use.

      ncolors = !D.TABLE_SIZE
      SET_PLOT, oldDevice
    ENDIF ELSE ncolors = !D.TABLE_SIZE
ENDIF
IF N_ELEMENTS(bottom) EQ 0 THEN bottom = 0B
IF N_ELEMENTS(charsize) EQ 0 THEN charsize = 1.0
IF N_ELEMENTS(format) EQ 0 THEN format = '(I5)'
IF N_ELEMENTS(color) EQ 0 THEN color = !P.Color
IF N_ELEMENTS(minrange) EQ 0 THEN minrange = 0
IF N_ELEMENTS(maxrange) EQ 0 THEN maxrange = ncolors
IF N_ELEMENTS(ticklen) EQ 0 THEN ticklen = 0.2
IF N_ELEMENTS(minor) EQ 0 THEN minor = 2
IF N_ELEMENTS(range) NE 0 THEN BEGIN
   minrange = range[0]
   maxrange = range[1]
ENDIF
IF N_ELEMENTS(divisions) EQ 0 THEN divisions = 6
IF N_ELEMENTS(font) EQ 0 THEN font = !P.Font
IF N_ELEMENTS(title) EQ 0 THEN title = ''

IF KEYWORD_SET(vertical) THEN BEGIN
   bar = REPLICATE(1B,20) # BINDGEN(ncolors)
   IF N_ELEMENTS(position) EQ 0 THEN BEGIN
      position = [0.88, 0.1, 0.95, 0.9]
   ENDIF ELSE BEGIN
      IF position[2]-position[0] GT position[3]-position[1] THEN BEGIN
         position = [position[1], position[0], position[3], position[2]]
      ENDIF
      IF position[0] GE position[2] THEN Message, "Position coordinates can't be reconciled."
      IF position[1] GE position[3] THEN Message, "Position coordinates can't be reconciled."
   ENDELSE
ENDIF ELSE BEGIN
   bar = BINDGEN(ncolors) # REPLICATE(1B, 20)
   IF N_ELEMENTS(position) EQ 0 THEN BEGIN
      position = [0.1, 0.88, 0.9, 0.95]
   ENDIF ELSE BEGIN
      IF position[3]-position[1] GT position[2]-position[0] THEN BEGIN
         position = [position[1], position[0], position[3], position[2]]
      ENDIF
      IF position[0] GE position[2] THEN Message, "Position coordinates can't be reconciled."
      IF position[1] GE position[3] THEN Message, "Position coordinates can't be reconciled."
   ENDELSE
ENDELSE

   ; Scale the color bar.

 bar = BYTSCL(bar, TOP=(ncolors-1 < (255-bottom))) + bottom

   ; Get starting locations in NORMAL coordinates.

xstart = position(0)
ystart = position(1)

   ; Get the size of the bar in NORMAL coordinates.

xsize = (position(2) - position(0))
ysize = (position(3) - position(1))

   ; Display the color bar in the window. Sizing is
   ; different for PostScript and regular display.

IF postScriptDevice THEN BEGIN

   TV, bar, xstart, ystart, XSIZE=xsize, YSIZE=ysize, /Normal

ENDIF ELSE BEGIN

   bar = CONGRID(bar, CEIL(xsize*!D.X_VSize), CEIL(ysize*!D.Y_VSize), /INTERP)

        ; Decomposed color off if device supports it.

   CASE  StrUpCase(!D.NAME) OF
        'X': BEGIN
            IF thisRelease GE 5.2 THEN Device, Get_Decomposed=thisDecomposed
            Device, Decomposed=0
            ENDCASE
        'WIN': BEGIN
            IF thisRelease GE 5.2 THEN Device, Get_Decomposed=thisDecomposed
            Device, Decomposed=0
            ENDCASE
        'MAC': BEGIN
            IF thisRelease GE 5.2 THEN Device, Get_Decomposed=thisDecomposed
            Device, Decomposed=0
            ENDCASE
        ELSE:
   ENDCASE

   TV, bar, xstart, ystart, /Normal

      ; Restore Decomposed state if necessary.

   CASE StrUpCase(!D.NAME) OF
      'X': BEGIN
         IF thisRelease GE 5.2 THEN Device, Decomposed=thisDecomposed
         ENDCASE
      'WIN': BEGIN
         IF thisRelease GE 5.2 THEN Device, Decomposed=thisDecomposed
         ENDCASE
      'MAC': BEGIN
         IF thisRelease GE 5.2 THEN Device, Decomposed=thisDecomposed
         ENDCASE
      ELSE:
   ENDCASE

ENDELSE

   ; Annotate the color bar.

IF KEYWORD_SET(vertical) THEN BEGIN

   IF KEYWORD_SET(right) THEN BEGIN

      PLOT, [minrange,maxrange], [minrange,maxrange], /NODATA, XTICKS=1, $
         YTICKS=divisions, XSTYLE=1, YSTYLE=9, $
         POSITION=position, COLOR=color, CHARSIZE=charsize, /NOERASE, $
         YTICKFORMAT='(A1)', XTICKFORMAT='(A1)', YTICKLEN=ticklen , $
         YRANGE=[minrange, maxrange], FONT=font, _EXTRA=extra, YMINOR=minor

      AXIS, YAXIS=1, YRANGE=[minrange, maxrange], YTICKFORMAT=format, YTICKS=divisions, $
         YTICKLEN=ticklen, YSTYLE=1, COLOR=color, CHARSIZE=charsize, $
         FONT=font, YTITLE=title, _EXTRA=extra, YMINOR=minor

   ENDIF ELSE BEGIN

      PLOT, [minrange,maxrange], [minrange,maxrange], /NODATA, XTICKS=1, $
         YTICKS=divisions, XSTYLE=1, YSTYLE=9, YMINOR=minor, $
         POSITION=position, COLOR=color, CHARSIZE=charsize, /NOERASE, $
         YTICKFORMAT=format, XTICKFORMAT='(A1)', YTICKLEN=ticklen , $
         YRANGE=[minrange, maxrange], FONT=font, YTITLE=title, _EXTRA=extra

      AXIS, YAXIS=1, YRANGE=[minrange, maxrange], YTICKFORMAT='(A1)', YTICKS=divisions, $
         YTICKLEN=ticklen, YSTYLE=1, COLOR=color, CHARSIZE=charsize, $
         FONT=font, _EXTRA=extra, YMINOR=minor

   ENDELSE

ENDIF ELSE BEGIN

   IF KEYWORD_SET(top) THEN BEGIN

      PLOT, [minrange,maxrange], [minrange,maxrange], /NODATA, XTICKS=divisions, $
         YTICKS=1, XSTYLE=9, YSTYLE=1, $
         POSITION=position, COLOR=color, CHARSIZE=charsize, /NOERASE, $
         YTICKFORMAT='(A1)', XTICKFORMAT='(A1)', XTICKLEN=ticklen, $
         XRANGE=[minrange, maxrange], FONT=font, _EXTRA=extra, XMINOR=minor

      AXIS, XTICKS=divisions, XSTYLE=1, COLOR=color, CHARSIZE=charsize, $
         XTICKFORMAT=format, XTICKLEN=ticklen, XRANGE=[minrange, maxrange], XAXIS=1, $
         FONT=font, XTITLE=title, _EXTRA=extra, XCHARSIZE=charsize, XMINOR=minor

   ENDIF ELSE BEGIN

      PLOT, [minrange,maxrange], [minrange,maxrange], /NODATA, XTICKS=divisions, $
         YTICKS=1, XSTYLE=1, YSTYLE=1, TITLE=title, $
         POSITION=position, COLOR=color, CHARSIZE=charsize, /NOERASE, $
         YTICKFORMAT='(A1)', XTICKFORMAT=format, XTICKLEN=ticklen, $
         XRANGE=[minrange, maxrange], FONT=font, XMinor=minor, _EXTRA=extra

    ENDELSE

ENDELSE

   ; Restore the previous plot and map system variables.

!P = bang_p
!X = bang_x
!Y = bang_y
!Z = bang_z
!Map = bang_map

END


pro create_movie
common choice, plotme
common parameters,nx,ny,tslices,xmax,ymax,twci,rhoi,teti,mime,wpewce,numq
common picdata, field, struct
common colortable,rgb,usecolor,red,blue,green,tmax,tmin,fooa1,fooa2,range1,range2,facmax,facmin,r1,r2,foob1,tmaxb,tminb
common controlplot,v
common imagesize,nxpix,nypix,xoff,yoff,xpic,ypic

; Temporary storage

fulldata = fltarr(nx,ny)
myimage = bytarr(nxpix,nypix)

; Open MPEG sequence

mpegID = MPEG_OPEN([nxpix,nypix],QUALITY=70)

; Set intital frame and number of padded frames

iframe = 0
ipad = 1
iskip = 1

;  Loop over all time slices

itime=0
for itime = 0,tslices,iskip do begin
;for itime = 0,400,iskip do begin
v.time = itime
doplot,2

; Add image in current window to MPEG movie

for i=1,ipad do begin
    iframe = iframe + 1
    MPEG_PUT, mpegID, WINDOW=!D.Window, FRAME=iframe, /ORDER
endfor

; end of loop

end

; SAVE the MPEG sequence

MPEG_SAVE, mpegID,FILENAME=strcompress(plotme(v.quantity)+'.mpg',/remove_all)

; Close the MPEG sequence

MPEG_CLOSE, mpegID

;  end of subroutine
end

pro viewer
common choice, plotme
common parameters,nx,ny,tslices,xmax,ymax,twci,rhoi,teti,mime,wpewce,numq
common picdata, field, struct
common colortable,rgb,usecolor,red,blue,green,tmax,tmin,fooa1,fooa2,range1,range2,facmax,facmin,r1,r2,foob1,tmaxb,tminb
common controlplot,v
common imagesize,nxpix,nypix,xoff,yoff,xpic,ypic

; Image size : Homa

nxpix=800
nypix=300

; Scaling factors for color contour plots

facmax=1.16
facmin=-0.05

; Declare structure for controlling the plots

v={quantity:0,time:0,ymin:0.0,ymax:0.0,yslice:0.0,xslice:0.0,xmin:0.0,xmax:0.0,smoothing:1,contours:24,plottype:0,overplot:0,shift:0,data:0,divide:0,multiply:0,map:0}

; Declare strings for menu

ptype = strarr(5)  

; Read in my color table

nc=256
rgb=fltarr(3,nc)
usecolor=fltarr(3,nc)
red=fltarr(nc)
blue=fltarr(nc)
green=fltarr(nc)
openr, unit, '~/Color_Tables/c5.tbl', /get_lun
;openr, unit, '~/Color_Tables/color.tbl', /get_lun
readf, unit, rgb
close, unit
free_lun, unit
red=rgb(0,*)
green=rgb(1,*)
blue=rgb(2,*)
tvlct,red,green,blue

; Declare two integers (long - 32 bit)

nx=500000
ny=500000
numq=500000

; Determine if there are multiple data directories and allow user to select

datafiles = file_search('*',count=numq) 
directory = dialog_pickfile(/directory,filter='*',TITLE='Choose directory with data'); else  directory = 'data/'
    
; open binary file for problem description 

if ( file_test(directory+'/info') eq 1 ) then begin
    print," *** Found Data Information File *** "
endif else begin
    print," *** Error - File info is missing ***"
endelse

;  First Try little endian then switch to big endian if this fails

little = 0
;on_ioerror, switch_endian
;openr,unit,directory+'/info', /get_lun
;readf,unit,nx,ny
;print,"nx = ",nx,"ny = ",ny
;little=1
;switch_endian: if (not little) then begin
;    print, " ** Little endian failed --> Switch to big endian"
;    close,unit
;    free_lun,unit
;    openr,unit,directory+'/info',/get_lun
;    readf,unit,nx,ny
;endif

; Read the problem desciption

;on_ioerror, halt_error1
;readf,unit,xmax,ymax
;readf,unit,toutput
nx=1800.
ny=1800.
;nz=3
xmax=128.
ymax=128.
zmax=31.415926
toutput=0.5

mime=1
wpewce=1.
rhoi=1.
teti=1.

;close,unit
;free_lun, unit

; Convert to ion skin depths


; Find the names of data files in the data directory

datafiles = file_search(directory+'/*.gda*',count=numq) 

; Now open each file and save the basename to identify later

print," Number of files=",numq
plotme = strarr(numq+1)  
instring='     '
plotme(0)='None'
for i=1,numq do begin
    if (not little) then openr,i,datafiles(i-1) else openr,i,datafiles(i-1),/swap_if_big_endian
    plotme(i) = file_basename(datafiles(i-1),'.gda')
    print,"i=",i," --> ",plotme(i)
endfor

; Close the input file

close,unit
free_lun, unit

; Define twci

twci = toutput/mime


; Echo information

print,'nx=',nx,' ny=',ny
print,'xmax/L=',xmax,' ymax/L=',ymax
print,' output every t*wci=',twci

; Association data to use direct access

; Homa - You have to modify this line for your data format - i.e. no
;        time or integer, just the data.   You also have to change any lines 
;       that reference struct.time or struct.it, since these don't
;       exist in your format

;struct = {data:fltarr(nx,ny),time:0.0,it:500000}
struct = {data:fltarr(nx,ny)}
field = assoc(1,struct)

; Determine number of time slices

information=file_info(datafiles(0))
tslices=information.size/(4*(nx*ny))-1
print,"File Size=",information.size
print,"Time Slices",tslices
if (tslices lt 1) then tslices=1

; Plot type

ptype(0)='Contour'
ptype(1)='Y-Average'
ptype(2)='Contour+Y-Average'
ptype(3)='Contour+Y-Slice'
ptype(4)='Contour+X-Slice'

; Options menu

desc =[ '1\Options' , $
        '1\Smoothing' , $
        '0\2' , $
        '0\3' , $
        '0\4' , $
        '0\5' , $
        '0\6' , $
        '0\7' , $
        '0\8' , $
        '2\off' , $
        '1\Contours' , $
        '0\12' , $
        '0\14' , $
        '0\16' , $
        '0\18' , $
        '0\20' , $
        '0\22' , $
        '0\24' , $
        '0\26' , $
        '0\28' , $
        '0\30' , $
        '0\32' , $
        '0\34' , $
        '0\36' , $
        '0\38' , $
        '0\40' , $
        '0\42' , $
        '0\44' , $
        '0\46' , $
        '2\48' , $
        '1\Transform Data' , $
        '0\Full Data' , $
        '0\Perturbed', $
        '2\',  $
        '1\Color Map', $
        '0\Default' , $
        '2\Load Table' ]

; Setup widgets

base = widget_base(/column)
row1 = widget_base(base,/row,scr_ysize=nypix/7)
row2 = widget_base(base,/row)
button1 = widget_button(row1, value = '  Done  ', uvalue = 'done',/sensitive)
button2 = widget_button(row1, value = '  Plot  ', uvalue = 'plot',/sensitive)
button3 = widget_button(row1, value = 'Render PS', uvalue = 'render')
button4 = widget_button(row1, value = 'Animate', uvalue = 'animate')
list5=widget_list(row1,value=ptype,uvalue='ptype')
list1=widget_droplist(row1,title=' Plot',value=plotme,uvalue='quantity')
list2=widget_droplist(row1,title=' Multiply',value=plotme,uvalue='multiply')
list3=widget_droplist(row1,title=' Divide',value=plotme,uvalue='divide')
list4=widget_droplist(row1,title=' Overplot',value=plotme,uvalue='overplot')
opt = cw_pdmenu(row1,desc,/return_index,uvalue='options')
slider2=cw_fslider(row2,title='Ymin',value=0.0,format='(f7.3)',uvalue='ymin',minimum=0.0,maximum=ymax,/drag)
slider3=cw_fslider(row2,title='Ymax',value=ymax,format='(f7.3)',uvalue='ymax',minimum=0.0,maximum=ymax,/drag)
slider6=cw_fslider(row2,title='Xmax',value=xmax,format='(f7.3)',uvalue='xmax',minimum=0.0,maximum=xmax,/drag)
slider4=cw_fslider(row2,title='Y-Slice',value=ymax,format='(f7.3)',uvalue='yslice',minimum=0.0,maximum=ymax,/drag,scroll=0.05)
slider5=cw_fslider(row2,title='X-Slice',value=0.0,format='(f7.3)',uvalue='xslice',minimum=xmin,maximum=xmax,/drag,scroll=0.05)
;slider7=widget_slider(row2,title=' Y-Shift',scrol=1,value=0,uvalue='shift',minimum=-ny/2,maximum=ny/2)
slider1=widget_slider(row2,title='Time Slice',scrol=1,value=0,uvalue='time',minimum=0,maximum=tslices,scr_xsize=nxpix/3.6)
draw = widget_draw(base,retain=2, xsize = nxpix, ysize = nypix,/button_events,uvalue='mouse')
widget_control, base, /realize
widget_control,list5,set_list_select=0
widget_control,list1,set_droplist=1
widget_control, base, set_uvalue={quantity:1,time:0,ymin:0.0,ymax:ymax,yslice:ymax,xslice:0.0,xmin:0,xmax:xmax,smoothing:2,contours:24,plottype:0,overplot:0,shift:0,data:0,divide:-1,multiply:-1,map:0}
widget_control, draw, get_value = index
xmanager, 'handle', base
halt_error1: print, ' *** Halting Program ***'
end

