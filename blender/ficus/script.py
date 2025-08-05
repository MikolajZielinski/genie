import bpy
import os

# Set output directory
output_dir = "blender/ficus/plys"  # Change this!

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Set the frame range
start_frame = bpy.context.scene.frame_start
end_frame = bpy.context.scene.frame_end

# Name of the object to export
object_names = ["ficus_main", "mask_ficus1_910k"]
for i in range(1, 561):
    if i == 12:
        continue
    object_names.append(f"mask_ficus1_910k.{i:03d}")

# Deselect everything
bpy.ops.object.select_all(action='DESELECT')

# Ensure it is selected and active
for name in object_names:
    obj = bpy.data.objects[name]
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

# Loop through each frame
for frame in range(start_frame, end_frame + 1):
    bpy.context.scene.frame_set(frame)
    bpy.context.view_layer.update()  # Ensure simulation is evaluated

    # Export to .obj
    filename = f"{(frame - 1):05d}.ply"
    filepath = os.path.join(output_dir, filename)

    bpy.ops.wm.ply_export(
        filepath=filepath,
        apply_modifiers=True,
        export_selected_objects=True,
        export_normals=True,
        export_uv=True,
        export_colors='NONE',        # options: 'NONE', 'SRGB', 'LINEAR'
        export_attributes=False,
        export_triangulated_mesh=False,
        ascii_format=True,          # Set to False for binary PLY
        forward_axis='Y',
        up_axis='Z'
    )

    print(f"Exported frame {frame} to {filepath}")