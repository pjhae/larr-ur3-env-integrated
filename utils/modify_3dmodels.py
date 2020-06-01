import glob
import os
import parse
import pybullet as p
import pybullet_data as pd

def convex_decomposition():
    p.connect(p.DIRECT)
    # 1. Duck (example)
    name_in = os.path.join(pd.getDataPath(), 'duck.obj')
    name_out = 'duck_vhacd2.obj'
    name_log = 'duck_vhacd2.log.txt'
    p.vhacd(name_in, name_out, name_log, alpha=0.04,resolution=1000000,depth=20,planeDownsampling=4,convexhullDownsampling=4,maxNumVerticesPerCH=256)
    # 2. Dual UR3 stand
    name_in = os.path.join(os.path.dirname(__file__), 'dual_ur3_stand_final_filtered.stl.obj')
    name_out = 'dual_ur3_stand_vhacd2.obj'
    name_log = 'dual_ur3_stand_vhacd2.log.txt'
    p.vhacd(name_in, name_out, name_log, alpha=0.04,resolution=32000000,depth=32,maxNumVerticesPerCH=256)

def split_obj_to_stls():
    # 1.
    # name_in = os.path.join(os.path.dirname(__file__), 'res64M_numvert1024', 'dual_ur3_stand_vhacd2.obj')
    # name_out = os.path.join(os.path.dirname(__file__), 'mujoco')
    # file_out = 'dual_ur3_stand_convex'
    # 2.
    name_in = os.path.join(os.path.dirname(__file__), 'dual_ur3_stand_vhacd2.obj')
    name_out = os.path.join(os.path.dirname(__file__), 'out')
    file_out = 'test_convex'

    f_format = 'f {:d} {:d} {:d} \n'
    
    first_run = True
    done = False
    num_out = 0
    vertices, faces = [], []
    with open(name_in, 'r') as fin:
        while not done:
            if first_run:
                try:
                    line = next(fin)
                except StopIteration:
                    done = True
            if 'o convex_' in line:
                first_run = False
                header = line
                with open('%s/%s%d.obj'%(name_out, file_out, num_out), 'w') as fout:
                    fout.write(header)
                    vcount, fcount = 0, 0
                    while not done:
                        try:
                            line = next(fin)
                        except StopIteration:
                            done = True
                            break
                        if 'o convex_' in line:
                            vertices.append(vcount)
                            faces.append(fcount)
                            break
                        elif 'v ' in line:
                            fout.write(line)
                            vcount += 1
                        elif 'f ' in line:
                            parsed = parse.parse(f_format, line)
                            line = f_format.format(parsed[0] - sum(vertices), parsed[1] - sum(vertices), parsed[2] - sum(vertices))
                            fout.write(line)
                            fcount += 1
                        else:
                            fout.write(line)
                os.system('meshlabserver -i %s/%s%d.obj -o %s/%s%d.stl'%(name_out, file_out, num_out, name_out, file_out, num_out))
                num_out += 1

def generate_xml_template():
    dir_in = os.path.join(os.path.dirname(__file__), 'out')
    stls = sorted(glob.glob('%s/*.stl'%(dir_in)))

    with open('%s/asset.xml'%dir_in, 'w') as fout:
        for stl in stls:
            fout.write('<mesh name="%s" file="ur3/dual_ur3_stand_collision_box/%s"></mesh>\n'%(stl, os.path.basename(stl)))
    
    with open('%s/geom.xml'%dir_in, 'w') as fout:
        for stl in stls:
            fout.write('<geom name="%s" pos="0 0 0" type="mesh" contype="1" conaffinity="0" group="0" rgba="0.2 0.2 0.2 0" mesh="%s"/>\n'%(stl, stl))
        
if __name__ == '__main__':
    # convex_decomposition()
    # split_obj_to_stls()
    generate_xml_template()
