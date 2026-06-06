from neuron import h
import numpy as np
import matplotlib.pyplot as plt


h.load_file("stdrun.hoc")
h.load_file("import3d.hoc")

class Cell:
    def __init__(self, swc_path, segment_length=20, Ra=100, cm=1, g_pas=1e-4, e_pas=-45):
        self.segment_length = segment_length
        self.Ra = Ra
        self.cm = cm
        self.g_pas = g_pas
        self.e_pas = e_pas
        
        self.load_morphology(swc_path)
        self.define_biophysics()
        self.insert_extracellular()
        self.verts, self.edges, self.radii = self.get_vertices_edges_radii()
        self.seg_xyz, self.seg_refs = self.get_segment_xyz()

    def load_morphology(self, swc_path):
        reader = h.Import3d_SWC_read()
        reader.input(swc_path)
        imprt = h.Import3d_GUI(reader, False)
        imprt.instantiate(self)
        h.define_shape()

    def define_biophysics(self):
        for sec in self.all:
            # this further subdivides branch-branch or branch-end 'sections' into smaller 
            # electrical segments called nseg subdivisions, 
            # roughly every 10-20 microns depending of section length  
            sec.nseg = 1 + 2 * int(sec.L / 20)  # TODO: is 20 microns the best?

            sec.Ra = self.Ra
            sec.cm = self.cm
            sec.insert("pas")
            sec.g_pas = self.g_pas
            sec.e_pas = self.e_pas

    def insert_extracellular(self):
        for sec in self.all:
            sec.insert("extracellular")
    
    # def get_segment_xyz(self):
    #     # just returns the center xyz of each segment
    #     # this will be used to compute the extracellular potential at each segment given some electric field
    #     seg_xyz = []
    #     seg_refs = []
    #     for sec in self.all:
    #         for seg in sec:
    #             x = seg.x
    #             xyz = [h.x3d(x, sec=sec), h.y3d(x, sec=sec), h.z3d(x, sec=sec)]
    #             seg_xyz.append(xyz)
    #             seg_refs.append(seg)
    #     return np.array(seg_xyz), seg_refs
    
    def get_segment_xyz(self):
        """
        Return the center xyz of each electrical segment.
        Coordinates are in microns.
        """
        # just returns the center xyz of each segment
        # this will be used to compute the extracellular potential at each segment given some electric field
        seg_xyz = []
        seg_refs = []

        for sec in self.all:
            n3d = int(sec.n3d())

            arc = np.array([sec.arc3d(i) for i in range(n3d)])
            xs = np.array([sec.x3d(i) for i in range(n3d)])
            ys = np.array([sec.y3d(i) for i in range(n3d)])
            zs = np.array([sec.z3d(i) for i in range(n3d)])

            for seg in sec:
                target_arc = seg.x * sec.L

                x = np.interp(target_arc, arc, xs)
                y = np.interp(target_arc, arc, ys)
                z = np.interp(target_arc, arc, zs)

                seg_xyz.append([x, y, z])
                seg_refs.append(seg)

        return np.array(seg_xyz), seg_refs
        
    def get_vertices_edges_radii(self):
        # pulling from neuron obj for debugging purposes instead of from swc
        verts = []
        edges = []
        radii = []
        sec_start_idx = {}

        # first pass: collect verts, radii, and edges in section
        for sec in self.all:
            start = len(verts)
            sec_start_idx[sec] = start

            n3d = int(sec.n3d())

            for i in range(n3d):
                verts.append([sec.x3d(i), sec.y3d(i), sec.z3d(i)])

                radii.append(sec.diam3d(i) / 2)


            for i in range(n3d - 1):
                edges.append([start + i, start + i + 1])

        # second pass: connect child sections to parent sections
        # has to be after ^ is built
        for sec in self.all:
            sr = h.SectionRef(sec=sec)

            if sr.has_parent():
                parent = sr.parent
                child_start = sec_start_idx[sec]

                parent_start = sec_start_idx[parent]
                parent_n3d = int(parent.n3d())

                child_xyz = np.array(verts[child_start])
                parent_xyz = np.array(verts[parent_start:parent_start + parent_n3d])

                closest_parent_local = np.argmin(
                    np.linalg.norm(parent_xyz - child_xyz, axis=1))

                closest_parent_global = parent_start + closest_parent_local
                edges.append([closest_parent_global, child_start])

        return (np.array(verts),
                np.array(edges, dtype=int),
                np.array(radii))

    def get_soma_sections(self):
        # pulls soma section objects
        soma_sections = [sec for sec in self.soma]
        return soma_sections


    def get_soma_centroid(self):
        # returns soma centroid in microns 
        soma_points = []

        for sec in self.get_soma_sections():
            n3d = int(sec.n3d())
            for point_idx in range(n3d):
                soma_points.append([
                    sec.x3d(point_idx),
                    sec.y3d(point_idx),
                    sec.z3d(point_idx),
                ])

        return np.mean(soma_points, dtype=float, axis = 0)

    def get_terminal_segment_indices(self, terminal_um=10.0, axis="z"):
        # pulls out a terminal proxy for class project # TODO once this is not class project, we have to have better indexing of terminals
        # last 10 microns 
        axis_to_idx = {"x": 0, "y": 1, "z": 2}

        coord_idx = axis_to_idx[axis]
        coords = self.seg_xyz[:, coord_idx]
        distal_edge = np.max(coords)
        threshold = distal_edge - float(terminal_um)

        terminal_indices = np.flatnonzero(coords >= threshold)
        if terminal_indices.size == 0:
            terminal_indices = np.array([int(np.argmax(coords))])

        return terminal_indices.astype(int)