"""Inspect VRM model structure."""
import struct, json, sys

for name in ['fem_vroid.vrm', 'masc_vroid.vrm']:
    path = f'data/models/{name}'
    with open(path, 'rb') as f:
        magic = f.read(4)
        ver = struct.unpack('<I', f.read(4))[0]
        length = struct.unpack('<I', f.read(4))[0]
        clen = struct.unpack('<I', f.read(4))[0]
        ctype = f.read(4)
        gltf = json.loads(f.read(clen))

    print(f'=== {name} ===')
    print(f'  Magic: {magic}, Version: {ver}, Size: {length}')
    print(f'  Meshes: {len(gltf.get("meshes", []))}')
    print(f'  Nodes: {len(gltf.get("nodes", []))}')
    print(f'  Skins: {len(gltf.get("skins", []))}')
    print(f'  Materials: {len(gltf.get("materials", []))}')
    print(f'  Images: {len(gltf.get("images", []))}')
    print(f'  Textures: {len(gltf.get("textures", []))}')

    ext = gltf.get('extensions', {})
    if 'VRM' in ext:
        vrm = ext['VRM']
        humanoid = vrm.get('humanoid', {})
        bones = humanoid.get('humanBones', [])
        print(f'  VRM 0.x humanoid bones: {len(bones)}')
        for b in bones:
            node_idx = b.get('node', -1)
            node_name = gltf['nodes'][node_idx].get('name', '?') if node_idx >= 0 else '?'
            print(f'    {b["bone"]:30s} -> node {node_idx} ({node_name})')
    elif 'VRMC_vrm' in ext:
        print('  VRM 1.0 format')
    else:
        print('  No VRM extension found')

    if gltf.get('skins'):
        skin = gltf['skins'][0]
        joints = skin.get('joints', [])
        print(f'  Skin joints: {len(joints)}')

    print()
