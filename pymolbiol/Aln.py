from Bio import AlignIO
from Bio.Phylo.TreeConstruction import DistanceCalculator
from Bio.Phylo.TreeConstruction import DistanceTreeConstructor
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import logomaker
import numpy as np
import pandas as pd
from copy import deepcopy
import re
import drawSvg as svg
import pickle

ambiguous_dna_values = {
    'A': 'A',
    'C': 'C',
    'G': 'G',
    'T': 'T',
    'M': 'AC',
    'R': 'AG',
    'W': 'AT',
    'S': 'CG',
    'Y': 'CT',
    'K': 'GT',
    'V': 'ACG',
    'H': 'ACT',
    'D': 'AGT',
    'B': 'CGT',
    'X': 'GATC',
    'N': 'GATC'}

aa_colors = {
    'A,I,L,M,F,W,V': '#80a0f0',
    'K,R': '#f01505',
    'E,D': '#c048c0',
    'N,Q,S,T':'#15c015',
    'C': '#f08080',
    'G': '#f09048',
    'P': '#c0c000',
    'H,Y': '#15a4a4',
    'X,-': '#ffffff',
    '*': '#000000'}
aa_colors = {k:v for kk,v in aa_colors.items() for k in kk if k != ','}


nt_colors = {
    'A': '#008000',
    'T,U': '#ff0000',
    'G': '#ffa500',
    'C': '#0000ff',
    'R,Y,M,K,W,H,B,D,V': '#808080',
    'N,X,-': '#ffffff',
    '*': '#000000'}
nt_colors = {k:v for kk,v in nt_colors.items() for k in kk if k != ','}

def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i+int(lv/3)], 16) for i in range(0, lv, int(lv/3)))

def rgb_to_hex(rgb):
    return '%02x%02x%02x' % rgb

def alpha_color(
    fg_color = '#ff0000', 
    fg_alpha = 0.3,
    bg_color = '#ffffff',
    bg_alpha = 1):
    """
    Provide fg_color+alpha to see what it becomes given the bg.
    """
    fg_color = np.array(hex_to_rgb(fg_color))/255
    bg_color = np.array(hex_to_rgb(bg_color))/255
    resulting_color = fg_color * fg_alpha + bg_color * bg_alpha * (1 - fg_alpha)
    return f'#{rgb_to_hex(tuple((resulting_color*255).astype(int)))}'

class Aln(MultipleSeqAlignment):
    def __init__(self, file=None, format = 'fasta'):
        """
        Supported inputs: 
            * pickle dump file, 
            * Bio.Align.MultipleSeqAlignment object, 
            * alignment files to open with AlignIO (FASTA by default).
        """
        if file is None:
            super(Aln, self).__init__([]) # empty instance
        elif isinstance(file, list):
            super(Aln, self).__init__(file) # record list object
        else:
            try:
                self = pickle.load(open(file, 'rb'))
            except:
                if isinstance(file, MultipleSeqAlignment): # MSA object
                    super(Aln, self).__init__([r for r in file])
                elif isinstance(file, str): # filename
                    super(Aln, self).__init__([r for r in AlignIO.read(file, format)])
                else:
                    raise RuntimeError('Input was\'nt recognized.')
        for r in self:
            r.name = r.id
            r.description = r.description[len(r.id):] if r.description.startswith(r.id) else r.description
    def nseq(self):
        return len(self)
    def npos(self):
        return self.get_alignment_length()
    def shape(self):
        return (self.nseq(), self.npos())
    def build_tree(self, start = None, end = None):
        calculator = DistanceCalculator('identity')
        dm = calculator.get_distance(self[:,start:end])
        constructor = DistanceTreeConstructor(calculator, 'upgma')
        tree = constructor.build_tree(self[:,start:end])
        tree_names = [str(k) for k in tree.depths().keys() if not re.match("^Inner[0-9]+$", str(k))]
        name2num = {self[i].id:i for i in range(len(self))}
        self = Aln([self[name2num[name]] for name in tree_names])
        self.seqrange_tree = [start, end]
        self.tree = tree
        return self
    def map_seqs(self, F = [], R = []):
        self.mapped_seqs = {'F':F, 'R':R}
    def __getitem__(self, i):
        got = super(Aln, self).__getitem__(i)
        if isinstance(got, MultipleSeqAlignment):
            return Aln(got)
        else:
            return got
    def __add__(self, other):
        return(Aln(super(Aln, self).__add__(other)))
    def show_logo(self):
        counts_mat = logomaker.alignment_to_matrix([str(r.seq) for r in self])
        logo = logomaker.Logo(counts_mat, figsize=[0.25 * self.npos(), 2])
        ticks0 = np.array([i for i in range(self.npos()) if ((i+1) % 5 == 0 and i < (self.npos())) or i == 0])
        ticks1 = ticks0 + 1
        logo.ax.set_xticks(ticks0)
        logo.ax.set_xticklabels(ticks1)
        logo.ax.set_yticks([0, 2.5, 5, 7.5, 10])
        logo.ax.set_yticklabels(['%0.2f'%i for i in [0, .25, .5, .75, 1]])
        logo.ax.set_ylabel('Freq')

    def draw(self, 
             scale=1,
             name_trans_fun = lambda seq_id, seq_descr: seq_id,
             show_alignment=True,
             show_letters=False,
             show_tree=True,
             show_plot=True,
             grey_mode='auto',
             no_color=False,
             freq_to_font_opacity=False,
             aln_type = 'nt', # or 'aa'
             plot_file='',
             plot_format='png',
             tree_width=1,
             uppercase=True,
             font_family='Menlo, Source Code Pro, Fira Mono, monospace',
             **kwargs):
        """
        Full plot has all these parts: tree, names, alignment, barplot.
        """    
        # set sizes and scales
        char_width = 5.5 * float(scale)
        char_box_size = 10 * scale
        seq_font_size = 10 * scale
        pos_font_size = 8 * scale
        name_font_size = 9 * scale
        text_xoffset = (char_box_size - char_width)/2 - char_width * 0.2
        text_yoffset = char_box_size * 0.15
        width_nudge = 0#.00001
        line_width = max(1, np.ceil(char_box_size/5))
        spacer = 2 * scale
        
        # N: names
        new_names = [name_trans_fun(r.id, r.description) for r in self]
        N_width_i = np.array([len(re.sub("<[^>]+?>", "", name)) for name in new_names]) * char_width
        N_width =  N_width_i.max() + spacer * 6
        # A: alignment
        A_width = self.npos() * char_box_size if show_alignment else 0
        A_pos_height = 1 * char_box_size  if show_alignment else 0
        A_seq_height = self.nseq() * char_box_size
        # T: tree
        if show_tree and hasattr(self, "tree") and scale >= .25:
            T_width = np.log2(self.npos()) * tree_width  * char_box_size
        else:
            T_width = 0
                
        # canvas
        canvas_width = N_width + A_width + T_width
        canvas_height = A_pos_height + A_seq_height
        svg_drawing = \
            svg.Drawing(width=canvas_width, 
                        height=canvas_height,
                        shape_rendering= "crispEdges", #"optimizeSpeed",
                        text_rendering="optimizeLegibility"
                              )
        # background color fill
        svg_object_list = \
            [svg.Rectangle(0,
                           0, 
                           canvas_width,
                           canvas_height,
                           fill="white")] #"#f5f5f5"
        
        # color mode 
        colors = nt_colors if aln_type == 'nt' else aa_colors # TODO: add pseudo-transparency (freq/mapping)
        colors = {k:alpha_color(v, .4) for k,v in colors.items()}
        grey_mode = True if ((grey_mode is True) or (grey_mode == 'auto' and self.npos() * self.nseq() > 1e6)) else False
        
        # draw tree
        if show_tree and hasattr(self, "tree"):
            # Arrays that store lines for the plot of clades 
            horizontal_linecollections = [] 
            vertical_linecollections = []
            text_labels = []

            # Layout 

            def get_x_positions(tree): 

                depths = tree.depths() 
                # If there are no branch lengths, assume unit branch lengths 
                if not max(depths.values()): 
                    depths = tree.depths(unit_branch_lengths=True) 
                return depths 

            def get_y_positions(tree): 
                maxheight = tree.count_terminals() 
                heights = {tip: maxheight - i 
                           for i, tip in enumerate(reversed(tree.get_terminals()))} 

                def calc_row(clade): 
                    for subclade in clade: 
                        if subclade not in heights: 
                            calc_row(subclade) 
                   # Closure over heights 
                    heights[clade] = (heights[clade.clades[0]] + 
                                heights[clade.clades[-1]]) / 2.0 

                if tree.root.clades: 
                    calc_row(tree.root) 
                return heights 
            
            x_posns = get_x_positions(self.tree)
            x_posns_max = max([v for k,v in x_posns.items()])
            x_posns = {k: (line_width/2 + v/x_posns_max*(T_width-line_width/2)) for k,v in x_posns.items()}
            y_posns = get_y_positions(self.tree)
            y_posns = {k:((self.nseq()-v+.3) * char_box_size + spacer) for k,v in y_posns.items()}

            def draw_clade_lines(orientation="horizontal", 
                             y_here=0, x_start=0, x_here=0, y_bot=0, y_top=0, 
                             color="black", 
                             stroke_dasharray = "1,0", lw=0): 
                if orientation == "horizontal": 
                    horizontal_linecollections.append(svg.Line( 
                        x_start, y_here, x_here, y_here, 
                        stroke=color, 
                        stroke_width=lw, 
                        stroke_dasharray=stroke_dasharray),) 
                elif orientation == "vertical": 
                    vertical_linecollections.append(svg.Line(
                        x_here, y_bot - lw/2, x_here, y_top + lw/2, 
                        stroke=color, 
                        stroke_width=lw, 
                        stroke_dasharray=stroke_dasharray),) 

            def draw_clade(clade, x_start, color, lw, label_func=str): 
                """Recursively draw a tree, down from the given clade.""" 
                is_leaf = False if re.match("^Inner[0-9]+$", str(clade)) else True
                x_here = x_posns[clade] if not is_leaf else T_width
                y_here = y_posns[clade]
                # Draw a horizontal line from start to here 
                draw_clade_lines(orientation="horizontal", 
                             y_here=y_here, x_start=x_start, x_here=x_here, color = "#dddddd" if is_leaf else color, 
                                 stroke_dasharray = f"{lw/2},{lw/2}" if is_leaf else "1,0", lw=lw) 
                # Add node/taxon labels 
                label = label_func(clade)

                if label not in (None, clade.__class__.__name__): 
                    text_labels.append(svg.Text(label, .2, x_here - .5, y_here),)

                if clade.clades: 
                   # Draw a vertical line connecting all children 
                    y_top = y_posns[clade.clades[0]] 
                    y_bot = y_posns[clade.clades[-1]] 
                    draw_clade_lines(orientation="vertical", 
                                x_here=x_here, y_bot=y_bot, y_top=y_top, color=color, lw=lw) 
                    # Draw descendents 
                    for child in clade: 
                        draw_clade(child, x_here, color, lw) 

            draw_clade(self.tree.root, 0, "#adadad", line_width) 
            svg_object_list.extend(horizontal_linecollections)
            svg_object_list.extend(vertical_linecollections)
            svg_object_list.extend(text_labels)
        
        for i in range(self.nseq()): #each seq
            # draw names
            svg_name_obj = \
                svg.Text(text = '', # we'll introduce the names later
                         fontSize = name_font_size, 
                         x = T_width + spacer * 3, # + text_xoffset,
                         y = (self.nseq()-i-1) * char_box_size + spacer, # + spacer
                         style=f"font-family: {font_family};",
                         textLength = N_width_i[i]) #  + spacer
            svg_name_obj.escapedText = new_names[i] # trick to include some markdown by name_trans_fun()
            svg_object_list += [svg_name_obj]
            
            # draw grey_mode rectangles
            if grey_mode:
            
                seq_letters = pd.Series(list(str(self[i,:].seq)))
                seq_colors = pd.Series([colors["*"] if letter != "-" else colors["-"] for letter in seq_letters])

                most_frequent_color = seq_colors.value_counts().index[0]
                seq_change_color = (seq_colors != seq_colors.shift())
                color_cell_len = seq_colors.groupby(seq_change_color.cumsum()).count()
                color_cell_len.index = seq_colors[seq_change_color].values
                left_side = T_width + N_width
                # draw a consensus rows first
                svg_object_list += \
                    [svg.Rectangle(x = left_side,
                                   y = A_seq_height - (i + 1) * char_box_size,
                                   height = char_box_size,
                                   width = A_width, 
                                   fill = most_frequent_color)]

                for color,length in color_cell_len.items(): # each pos
                    delta_x = length * char_box_size
                    if color != most_frequent_color:
                        svg_object_list += \
                            [svg.Rectangle(x = left_side,
                                           y = A_seq_height - (i + 1) * char_box_size,
                                           height = char_box_size,
                                           width = delta_x, 
                                           fill = color)]
                    left_side += delta_x
            
            # draw matched sequence arrows/rectangles
            
        # draw alignment
        if not hasattr(self, "seqrange_tree"):
            self.seqrange_tree = [0, self.npos()]
        
        seqrange_tree_start = (0 if self.seqrange_tree[0] is None else self.seqrange_tree[0])
        seqrange_tree_end = (self.npos() if self.seqrange_tree[1] is None else self.seqrange_tree[1])
        seqrange_tree_width = seqrange_tree_end - seqrange_tree_start
        svg_object_list += \
                    [svg.Rectangle(x = T_width + N_width + seqrange_tree_start * char_box_size,
                                   y = A_seq_height,
                                   height = char_box_size,
                                   width = char_box_size * seqrange_tree_width, 
                                   fill = "#dddddd")]
        for i in range(self.npos()): # each pos
            # draw position numbers
            if (i+1) % 5 == 0 and i < (self.npos()-1):
                svg_object_list += \
                    [svg.Text(text = str(i+1),
                              fontSize = pos_font_size, 
                              x = T_width + N_width + i * char_box_size + text_xoffset,
                              y = self.nseq() * char_box_size + 0.5 * scale + spacer, 
                              style = f"font-family: {font_family};")]
            # draw rectangles for position columns
            if not grey_mode:
                pos_letters = self[:,i] # TODO change indexing
                pos_colors = pd.Series([colors.get(letter, '#ffffff') for letter in pos_letters])
                most_frequent_color = pos_colors.value_counts().index[0]
                pos_change_color = (pos_colors != pos_colors.shift())
                color_cell_len = pos_colors.groupby(pos_change_color.cumsum()).count()
                color_cell_len.index = pos_colors[pos_change_color].values
                top_level = A_seq_height
                # draw a consensus letter first
                consensus = pos_colors
                svg_object_list += \
                    [svg.Rectangle(x = T_width + N_width + i * char_box_size,
                                   y = 0,
                                   height = A_seq_height,
                                   width = char_box_size * (1 + width_nudge), 
                                   fill = most_frequent_color)]

                for color,length in color_cell_len.items(): # each seq
                    delta_y = (length + width_nudge) * char_box_size
                    if color != most_frequent_color:
                        if no_color:
                            color = '#ffffff'
                        svg_object_list += \
                            [svg.Rectangle(x = T_width + N_width + i * char_box_size ,
                                           y = top_level - delta_y,
                                           height = delta_y,
                                           width = char_box_size * (1 + width_nudge), 
                                           fill = color)]
                    top_level -= delta_y
        if show_letters:
            for j in range(self.nseq()):
                for i in range(self.npos()):
                    text = self[j, i].upper() if uppercase else self[j, i]
                    svg_object_list += \
                        [svg.Text(text = text,
                                  fontSize = seq_font_size, 
                                  x = T_width + N_width + i * char_box_size + text_xoffset,
                                  y = (self.nseq()-j-1) * char_box_size + text_yoffset,
                                  style = f"font-family: {font_family};")]
        
        svg_drawing.extend(svg_object_list)
        
        if plot_file.endswith('.png') or plot_format == 'png':
            if plot_file != '':
                svg_drawing.savePng(plot_file)
            if show_plot:
                return(svg_drawing.rasterize())
        elif plot_file.endswith('.svg') or plot_format == 'svg':
            if plot_file != '':
                svg_drawing.saveSvg(plot_file)
            if show_plot:
                return(svg_drawing)


    def deduplicate(self, limits = []):
        '''
        There are two types of duplicates:
        same record twice (with same id) or
        several records with (almost) identical sequences.
        We can check identity within given limits
        (like [1, 4], starting from 1, inclusive interval )
        '''

        from Bio.Phylo.TreeConstruction import DistanceCalculator
        from collections import Counter
        import copy

        aln_names_orig = [rec.id for rec in self]

        # remove duplicate names
        aln_dict = {rec.id: rec for rec in self}
        self = Aln(list(aln_dict.values()))

        # remove identical sequences
        if len(limits) == 2:
            aln_region = self[:, (limits[0]-1):limits[1]]
        else:
            aln_region = self
        calculator = DistanceCalculator('identity')
        dm = calculator.get_distance(aln_region)

        selected_record_names = copy.deepcopy(dm.names)

        for rec_ix1 in range(0, len(dm.names)-1):
            for rec_ix2 in range(rec_ix1+1, len(dm.names)):
                #print(f'rec_ix1={rec_ix1}, rec_ix2={rec_ix2} len(dm.names)={len(dm.names)}')
                if dm[rec_ix1, rec_ix2] == 0:
                    record_name_to_delete = dm.names[rec_ix2]
                    if record_name_to_delete in selected_record_names:
                        selected_record_names.remove(record_name_to_delete)

        self = Aln([rec for rec in self if rec.id in selected_record_names])

        duplicate_ids = list(Counter(aln_names_orig) - Counter(selected_record_names))
        if len(duplicate_ids)>0:
            print('Following records were identified as duplicates:\n' + ', '.join(duplicate_ids) + '\n')

        return(self)

    # ============================================================ #

    def remove_gappy_columns(self):
        '''
        After throwing out some records some positions may contain
        only gaps. This function removes those columns (positions).
        '''

        aln_length = self.get_alignment_length()
        current_pos = 0
        aln = self

        while current_pos < aln_length:
            if len(aln[:,current_pos].replace('-', '').replace('!', '').replace('.', '').replace(' ', '')) == 0:
                aln = aln[:, :current_pos] + aln[:, current_pos+1:]
                aln_length -= 1
            else:
                current_pos += 1
        return(Aln(aln))

    # ============================================================ #

    def remove_ids(self, id_regex_list):
        '''
        Removes all records which names are encoded in regexps.
        '''

        import re

        id_regex_list = [recid.replace('(', '\(').replace(')', '\)') for recid in id_regex_list]
        id_regex_combined = '|'.join(id_regex_list)
        recs_to_keep = [r for r in self if not re.search(id_regex_combined, r.id)]
        recids_to_delete = [r.id for r in self if re.search(id_regex_combined, r.id)]
        print('Following records were removed:\n' + ', '.join(recids_to_delete) + '\n')
        result = Aln(recs_to_keep)
        return(result)

    # ============================================================ #

    def upper(self):
        '''
        Transforms all sequences in the MSA to upper case.
        '''

        from Bio.Seq import Seq

        for rec in self:
            rec.seq = Seq(str(rec.seq).upper())
        return(self)

    def to_dotted(self, fill='·', replace_gaps=False):
    
        from Bio.Seq import Seq

        query = self[0,:]
        recs = self[1:,:]
        for rec in recs:
            for pos in range(recs.get_alignment_length()):
                if (rec[pos] == query[pos] and not (rec[pos] == '-' and not replace_gaps)) \
                    or (replace_gaps and rec[pos] == '-'):
                    rec.seq = rec.seq[:pos] + Seq(fill) + rec.seq[pos+1:]
        aln = Aln([query] + list(recs))
        return(aln)