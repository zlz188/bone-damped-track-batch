bl_info = {
    "name": "Bone Damped Track Batch",
    "author": "zlz",
    "version": (3, 6),
    "blender": (4, 5, 0),
    "location": "3D Viewport N Panel > Bone Tools",
    "description": "Batch-add Damped Track constraints with gradient influence",
    "category": "Animation",
    "wiki_url": "https://github.com/zlz188/bone-damped-track-batch",
    "tracker_url": "https://github.com/zlz188/bone-damped-track-batch/issues",
}

import bpy
import blf
import gpu
from mathutils import Vector
from bpy_extras import view3d_utils
from gpu_extras.batch import batch_for_shader

# Constraints created by this add-on share this prefix, so they can be identified, updated and cleaned safely.
CONSTRAINT_PREFIX = "DTrack_"


def get_bone_and_descendants(armature, bone_names_set):
    """递归获取给定骨骼集合的所有后代骨骼名称"""
    result = set(bone_names_set)
    children_map = {}
    for bone in armature.data.bones:
        if bone.parent:
            children_map.setdefault(bone.parent.name, set()).add(bone.name)

    to_process = list(bone_names_set)
    while to_process:
        current = to_process.pop()
        if current in children_map:
            for child in children_map[current]:
                if child not in result:
                    result.add(child)
                    to_process.append(child)
    return result


def get_selected_bone_names(context):
    """根据当前模式获取选中的骨骼名称集合"""
    if context.mode == 'POSE':
        return {bone.name for bone in context.selected_pose_bones}
    elif context.mode == 'EDIT_ARMATURE':
        return {bone.name for bone in context.selected_bones}
    else:
        return set()


def compute_depths(bones):
    """
    自根向下按拓扑顺序计算层级深度，避免依赖骨骼遍历顺序。
    修复：原实现若子骨骼先于父骨骼被遍历，会以 0 打底算出错误深度，
    导致递增/递减渐变静默出错。

    返回 (depth_map, chain_max_map)：
    - depth_map:    骨骼名 -> 深度（该链第一个被处理的骨骼为 0）
    - chain_max_map:骨骼名 -> 所在链条的最大深度（用于"按链条比例"渐变，
      使梯度能按链条总长度均匀分布，不受链条长短影响）
    """
    bone_names = {b.name for b in bones}
    children_map = {}
    for b in bones:
        children_map.setdefault(b.name, [])
        if b.parent and b.parent.name in bone_names:
            children_map.setdefault(b.parent.name, []).append(b.name)

    # 种子：父骨骼不在处理范围内的骨骼（即各分支的根），深度为 0
    seeds = [b.name for b in bones
             if not (b.parent and b.parent.name in bone_names)]

    depth_map = {}
    chain_max_map = {}
    for seed in seeds:
        chain_names = []
        stack = [(seed, 0)]
        while stack:
            name, depth = stack.pop()
            depth_map[name] = depth
            chain_names.append(name)
            for child in children_map.get(name, []):
                stack.append((child, depth + 1))
        chain_max = max((depth_map[n] for n in chain_names), default=0)
        for n in chain_names:
            chain_max_map[n] = chain_max

    return depth_map, chain_max_map


def is_plugin_constraint(c):
    """判断约束是否为本插件创建的阻尼追踪约束"""
    return c.type == 'DAMPED_TRACK' and c.name.startswith(CONSTRAINT_PREFIX)


def get_pose_green_bone_names(arm_obj):
    """
    获取姿态模式下显示为"绿色"的骨骼名称（即已有约束的布料/头发骨骼）。

    Blender 姿态模式状态色逻辑（复现 Blender 内部判断，不读屏幕像素）：
    - 绿色：骨骼带有约束（任意约束，排除 IK Solver）
    - 黄色：IK Solver 约束（有约束但显示黄色，需排除）
    - 橙色：无目标、无法解算的约束（有约束但显示橙色，需排除）
    故：有约束 且 不含 IK 且 不含无目标约束 → 绿色。

    注意：
    - 本插件本身会添加约束，而"绿色"= 有约束。因此勾选后仅处理
      "原本已有约束"的骨骼，灰色无约束骨骼保持灰色。
    - 若骨骼启用了自定义骨骼颜色（use_custom_color），状态色会被覆盖，
      数据判断结果可能与视图颜色不一致。
    """
    green = set()
    for pb in arm_obj.pose.bones:
        if not pb.constraints:
            continue
        is_green = True
        for c in pb.constraints:
            if c.type == 'IK':
                is_green = False  # IK Solver 显示黄色
                break
            if hasattr(c, 'target') and c.target is None:
                is_green = False  # 无目标解算约束显示橙色
                break
        if is_green:
            green.add(pb.name)
    return green


def get_keyword_bone_names(arm_obj, keywords):
    """
    按关键词筛选布料/头发骨骼：骨骼名、MMD 日文名（mmd_bone_name_j）、
    MMD 英文名（mmd_bone_name_e）任一包含某关键词即命中（英文不区分大小写）。
    比"绿色骨骼"更精准——腿部等带约束的非布料骨骼只要名字不含关键词就不会被命中。
    """
    if not keywords:
        return set()
    kws = [k.lower() for k in keywords]

    matched = set()
    for pb in arm_obj.pose.bones:
        names = [pb.name]
        # mmd_tools 提供的 MMD 原始日文/英文骨骼名（未安装 mmd_tools 时 getattr 返回 ''）
        for prop in ('mmd_bone_name_j', 'mmd_bone_name_e'):
            val = getattr(pb, prop, '')
            if isinstance(val, str) and val:
                names.append(val)

        haystack = [n.lower() for n in names]
        if any(kw in h for h in haystack for kw in kws):
            matched.add(pb.name)
    return matched


def collect_bones_to_process(arm_obj, context, scene):
    """
    根据作用范围与过滤设置，计算将被处理的骨骼（Bone）列表。
    返回 (bones, need_pose_mode)；若 SELECTED 但无选中骨骼，返回 (None, False)。
    供"批量添加"与"预览命中"共用，保证两者判定完全一致。
    """
    scope = scene.bone_dt_scope
    skip_root = scene.bone_dt_skip_root
    filter_mode = scene.bone_dt_filter_mode

    need_pose_mode = False
    target_bone_names = None
    if scope == 'SELECTED':
        selected_names = get_selected_bone_names(context)
        if not selected_names:
            return None, False
        target_bone_names = get_bone_and_descendants(arm_obj, selected_names)
        need_pose_mode = (context.mode != 'POSE')

    filter_names = None
    if scope == 'ALL' and filter_mode != 'NONE':
        if filter_mode == 'GREEN':
            filter_names = get_pose_green_bone_names(arm_obj)
        elif filter_mode == 'KEYWORD':
            kws = [k.strip() for k in scene.bone_dt_keywords.split(',') if k.strip()]
            filter_names = get_keyword_bone_names(arm_obj, kws)

    # 手动补入/排除：按名称关键词（不区分大小写）在过滤结果之上做增删
    manual_include = [k.strip().lower() for k in scene.bone_dt_manual_include.split(',') if k.strip()]
    manual_exclude = [k.strip().lower() for k in scene.bone_dt_manual_exclude.split(',') if k.strip()]

    result = []
    for bone in arm_obj.data.bones:
        if scope == 'SELECTED' and bone.name not in target_bone_names:
            continue
        if skip_root and bone.parent is None:
            continue
        if not bone.children:  # 叶子骨骼无子可追，跳过
            continue

        name_l = bone.name.lower()

        # 手动排除：名称含任一排除关键词 → 剔除（优先级最高，即使被过滤命中）
        if manual_exclude and any(k in name_l for k in manual_exclude):
            continue

        # 过滤命中检查：未命中时，仅当名称含"手动补入"关键词才保留（补回被漏掉的骨骼）
        if filter_names is not None and bone.name not in filter_names:
            if not (manual_include and any(k in name_l for k in manual_include)):
                continue

        result.append(bone)
    return result, need_pose_mode


class BONE_DT_AddDampedTrackToBones(bpy.types.Operator):
    """批量添加阻尼追踪约束（支持影响值递增/递减）"""

    bl_idname = "bone_dt.add_damped_track_to_bones"
    bl_label = "Batch Add Damped Track Constraints"
    bl_description = "Add Damped Track constraints to the matched bones, with gradient influence support"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == 'ARMATURE'

    def execute(self, context):
        arm_obj = context.active_object
        scene = context.scene

        influence_base = scene.bone_dt_influence
        skip_root = scene.bone_dt_skip_root
        scope = scene.bone_dt_scope
        gradient_mode = scene.bone_dt_gradient_mode
        step = scene.bone_dt_gradient_step
        gradient_end = scene.bone_dt_gradient_end
        gradient_curve = scene.bone_dt_gradient_curve
        overwrite = scene.bone_dt_overwrite
        target_space = scene.bone_dt_target_space
        child_mode = scene.bone_dt_child_mode
        filter_mode = scene.bone_dt_filter_mode

        # 计算将被处理的骨骼（复用筛选逻辑，与"预览命中"完全一致）
        bones_to_process, need_pose_mode = collect_bones_to_process(arm_obj, context, scene)
        if bones_to_process is None:
            self.report({'ERROR'}, "Select at least one bone in Pose or Edit Mode first")
            return {'CANCELLED'}

        current_mode = context.mode
        try:
            # 仅在需要读取姿态选择时切换模式；ALL 范围下直接改 pose 数据即可
            if need_pose_mode:
                bpy.ops.object.mode_set(mode='POSE')

            depth_map, chain_max_map = compute_depths(bones_to_process)

            pose_bones = arm_obj.pose.bones
            processed = []
            for bone in bones_to_process:
                children = list(bone.children)
                if child_mode == 'LAST':
                    target_bone = children[-1]
                else:
                    target_bone = children[0]

                pose_bone = pose_bones.get(bone.name)
                if not pose_bone:
                    continue

                if gradient_mode == 'NONE':
                    influence = influence_base
                elif gradient_mode == 'INCREASE':
                    influence = influence_base + depth_map.get(bone.name, 0) * step
                    influence = max(0.0, min(1.0, influence))
                elif gradient_mode == 'DECREASE':
                    influence = influence_base - depth_map.get(bone.name, 0) * step
                    influence = max(0.0, min(1.0, influence))
                else:  # PROPORTIONAL 按链条比例：从起始值沿链条渐变到末端影响值
                    depth = depth_map.get(bone.name, 0)
                    chain_max = chain_max_map.get(bone.name, 0)
                    t = depth / chain_max if chain_max > 0 else 0.0
                    # 缓动曲线：把链条位置 t (0~1) 映射为渐变进度 s (0~1)
                    if gradient_curve == 'EASE_IN':
                        s = t * t  # 根部变化慢，末梢变化快
                    elif gradient_curve == 'EASE_OUT':
                        s = 1.0 - (1.0 - t) * (1.0 - t)  # 根部变化快，末梢变化缓
                    elif gradient_curve == 'SMOOTH':
                        s = t * t * (3.0 - 2.0 * t)  # S形过渡
                    else:  # LINEAR
                        s = t
                    influence = influence_base + (gradient_end - influence_base) * s
                    influence = max(0.0, min(1.0, influence))

                # 仅复用本插件创建的约束，避免误改用户手调的 Damped Track
                existing = next(
                    (c for c in pose_bone.constraints if is_plugin_constraint(c)),
                    None,
                )

                if existing:
                    if overwrite:
                        existing.target = arm_obj
                        existing.subtarget = target_bone.name
                        existing.target_space = target_space
                        existing.influence = influence
                        processed.append(bone.name)
                    # 不勾选覆盖时保持该约束原样
                else:
                    c = pose_bone.constraints.new('DAMPED_TRACK')
                    c.name = f"{CONSTRAINT_PREFIX}{target_bone.name}"
                    c.target = arm_obj
                    c.subtarget = target_bone.name
                    c.target_space = target_space
                    c.track_axis = 'TRACK_Y'
                    c.influence = influence
                    processed.append(bone.name)

            if processed:
                if filter_mode != 'NONE':
                    label = {'GREEN': 'pose-green bones', 'KEYWORD': 'cloth/hair bones'}.get(filter_mode, 'target bones')
                    self.report({'INFO'}, f"Matched {len(bones_to_process)} {label}, added/updated {len(processed)} constraints")
                else:
                    self.report({'INFO'}, f"Added/updated constraints on {len(processed)} bones")
            else:
                self.report({'WARNING'}, "No bones matched the current settings")
        finally:
            if need_pose_mode and current_mode != 'POSE':
                bpy.ops.object.mode_set(mode=current_mode)

        return {'FINISHED'}


class BONE_DT_PreviewMatchedBones(bpy.types.Operator):
    """预览：在姿态模式选中当前设置将命中的骨骼（不添加、不改动任何约束）"""

    bl_idname = "bone_dt.preview_matched_bones"
    bl_label = "Preview Matched Bones"
    bl_description = "Select (highlight) the bones the current settings would affect, without modifying anything"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == 'ARMATURE'

    def execute(self, context):
        arm_obj = context.active_object
        scene = context.scene
        filter_mode = scene.bone_dt_filter_mode

        bones_to_process, need_pose_mode = collect_bones_to_process(arm_obj, context, scene)
        if bones_to_process is None:
            self.report({'ERROR'}, "Select at least one bone in Pose or Edit Mode first")
            return {'CANCELLED'}

        matched = [b.name for b in bones_to_process]
        matched_set = set(matched)

        # 进入姿态模式并选中命中骨骼，便于在视口中高亮预览
        current_mode = context.mode
        try:
            if current_mode != 'POSE':
                bpy.ops.object.mode_set(mode='POSE')
            bpy.ops.pose.select_all(action='DESELECT')
            for pb in arm_obj.pose.bones:
                # Blender 5.0 起选择状态存放在 PoseBone.select；
                # 旧版本（4.x 及更早）存放在 Bone.select，需兼容处理
                if hasattr(pb, 'select'):
                    pb.select = pb.name in matched_set
                else:
                    pb.bone.select = pb.name in matched_set
        finally:
            if current_mode != 'POSE':
                bpy.ops.object.mode_set(mode=current_mode)

        if matched:
            label = {'GREEN': 'pose-green bones', 'KEYWORD': 'cloth/hair bones'}.get(filter_mode, 'target bones')
            self.report({'INFO'}, f"Matched {len(matched)} {label}, selected in Pose Mode")
            print(f"[DampedTrack] Matched {len(matched)} {label}: {', '.join(sorted(matched))}")
        else:
            self.report({'WARNING'}, "No bones matched; check the filter or keywords")
        return {'FINISHED'}


def _tag_view3d_redraw(context, fallback_area=None):
    """强制所有 3D 视口区域重绘，确保模态覆盖层实时刷新"""
    areas = []
    if fallback_area is not None:
        areas.append(fallback_area)
    screen = getattr(context, 'screen', None)
    if screen is not None:
        for area in screen.areas:
            if area.type == 'VIEW_3D':
                areas.append(area)
    seen = set()
    for area in areas:
        if id(area) in seen:
            continue
        seen.add(id(area))
        area.tag_redraw()


def _find_view3d_region(context):
    """从窗口区域中找一个带 region_data 的 3D 视口 WINDOW 区域，供绘制回调投影用"""
    screen = getattr(context, 'screen', None)
    if not screen:
        return None, None
    for area in screen.areas:
        if area.type != 'VIEW_3D':
            continue
        for region in area.regions:
            if region.type == 'WINDOW' and getattr(region, 'data', None) is not None:
                return region, region.data
    return None, None


def _get_uniform_shader():
    """
    获取 2D 像素绘制的单色着色器（带缓存）。
    Blender 5.0 起内置着色器整合为 UNIFORM_COLOR（POST_PIXEL 像素坐标直接传入）；
    4.x 及更早使用 2D_UNIFORM_COLOR，依次回退以兼容不同版本。
    """
    if _uniform_shader_cache[0] is not None:
        return _uniform_shader_cache[0]
    for name in ('UNIFORM_COLOR', '2D_UNIFORM_COLOR'):
        try:
            s = gpu.shader.from_builtin(name)
            _uniform_shader_cache[0] = s
            return s
        except Exception:
            continue
    return None


_uniform_shader_cache = [None]


def _draw_thick_line(shader, p1, p2, half_w, color):
    """在屏幕空间画一条指定半宽的粗线（四边形），用于高亮悬停骨骼"""
    d = p2 - p1
    if d.length < 1e-6:
        return
    perp = Vector((-d.y, d.x)).normalized() * half_w
    a = (p1 + perp).xy
    b = (p1 - perp).xy
    c = (p2 - perp).xy
    e = (p2 + perp).xy
    tris = [a, b, c, a, c, e]
    batch = batch_for_shader(shader, 'TRIS', {"pos": tris})
    shader.bind()
    shader.uniform_float("color", color)
    batch.draw(shader)


def _draw_square(shader, c, half, color):
    """在屏幕空间画一个填充方块，用于高亮骨骼端点"""
    tris = [
        (c.x - half, c.y - half), (c.x - half, c.y + half), (c.x + half, c.y + half),
        (c.x - half, c.y - half), (c.x + half, c.y + half), (c.x - half, c.y + half),
    ]
    batch = batch_for_shader(shader, 'TRIS', {"pos": tris})
    shader.bind()
    shader.uniform_float("color", color)
    batch.draw(shader)


class BONE_DT_PickBoneOperator(bpy.types.Operator):
    """吸管：在视图中点击骨骼，将其名称填入指定的手动补入/排除字段"""

    bl_idname = "bone_dt.pick_bone"
    bl_label = "Pick Bone (Eyedropper)"
    bl_description = "Click a bone in the viewport to add its name to the include/exclude field"
    bl_options = {'REGISTER'}

    target_prop: bpy.props.StringProperty(default="bone_dt_manual_include")

    _handle = None
    _hover = None
    _mx = 0
    _my = 0
    _orig_mode = None
    _area = None

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == 'ARMATURE'

    def invoke(self, context, event):
        self._orig_mode = context.mode
        if context.mode != 'POSE':
            bpy.ops.object.mode_set(mode='POSE')
        self._hover = None
        self._mx = event.mouse_region_x
        self._my = event.mouse_region_y
        self._area = context.area
        self._handle = bpy.types.SpaceView3D.draw_handler_add(
            self.draw_callback, (self, context), 'WINDOW', 'POST_PIXEL')
        context.window_manager.modal_handler_add(self)
        self._region = None
        self._rv3d = None
        if context.window:
            context.window.cursor_set('EYEDROPPER')
        self.report({'INFO'}, "Pick bone: hover to highlight, LMB to pick, RMB/Esc to cancel")
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type in {'RIGHTMOUSE', 'ESC'}:
            return self._finish(context, cancelled=True)

        # 每次事件都强制所有 3D 视口重绘，让覆盖层（十字准星/高亮）跟随鼠标实时刷新
        _tag_view3d_redraw(context, self._area)

        if event.type == 'MOUSEMOVE':
            self._mx = event.mouse_region_x
            self._my = event.mouse_region_y
            self._region = context.region
            self._rv3d = context.region_data
            self._hover = self._find_bone_at(context, event)

        elif event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            name = self._find_bone_at(context, event)
            if name:
                self._add_to_field(context, name)
                self.report({'INFO'}, f"Picked bone: {name}")
                return self._finish(context, cancelled=False)
            self.report({'WARNING'}, "No bone picked; click a bone body or its endpoint")

        return {'RUNNING_MODAL'}

    def _find_bone_at(self, context, event):
        """把每根骨骼的头/尾投影到屏幕，返回离鼠标最近的骨骼名"""
        region = context.region
        rv3d = context.region_data
        if not (region and rv3d):
            return None
        arm = context.active_object
        mx = event.mouse_region_x
        my = event.mouse_region_y
        best = None
        best_d = 20.0
        for pb in arm.pose.bones:
            for pt in (arm.matrix_world @ pb.head, arm.matrix_world @ pb.tail):
                v = view3d_utils.location_3d_to_region_2d(region, rv3d, pt)
                if v is None:
                    continue
                d = (Vector((v.x, v.y)) - Vector((mx, my))).length
                if d < best_d:
                    best_d = d
                    best = pb.name
        return best

    def _add_to_field(self, context, name):
        """把拾取到的骨骼名写入目标字段（去重、逗号分隔）"""
        scene = context.scene
        field = getattr(scene, self.target_prop)
        names = [n.strip() for n in field.split(',') if n.strip()]
        if name not in names:
            names.append(name)
            setattr(scene, self.target_prop, ", ".join(names))

    def _finish(self, context, cancelled):
        if self._handle is not None:
            bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
            self._handle = None
        if context.window:
            context.window.cursor_set('DEFAULT')
        if self._orig_mode != 'POSE' and context.mode == 'POSE':
            try:
                bpy.ops.object.mode_set(mode=self._orig_mode)
            except Exception:
                pass
        # 移除绘制回调后立即强制重绘一次，清除滞留的十字准星/高亮/名字残影
        _tag_view3d_redraw(context, self._area)
        if not cancelled:
            # 拾取完成后自动刷新一次"预览命中"，让视图里的命中高亮即时更新
            try:
                bpy.ops.bone_dt.preview_matched_bones()
            except Exception:
                pass
        return {'CANCELLED'} if cancelled else {'FINISHED'}

    def draw_callback(self, op, context):
        x, y = self._mx, self._my
        # 优先使用模态中实时捕获的视口 region（invoke 时冻结的 context 是侧栏，region_data 为空）
        region = self._region
        rv3d = self._rv3d
        if not (region and rv3d):
            region, rv3d = _find_view3d_region(context)

        # 高亮/准星用 GPU 绘制；5.0 起着色器名为 UNIFORM_COLOR，失败时静默跳过 GPU 部分
        try:
            shader = _get_uniform_shader()
            if shader is not None:
                # 红色高亮当前悬停的骨骼（粗线 + 端点方块），与预览的选中高亮（橙/白）区分
                if self._hover and region and rv3d:
                    arm = context.active_object
                    pb = arm.pose.bones.get(self._hover)
                    if pb:
                        p1 = view3d_utils.location_3d_to_region_2d(region, rv3d, arm.matrix_world @ pb.head)
                        p2 = view3d_utils.location_3d_to_region_2d(region, rv3d, arm.matrix_world @ pb.tail)
                        if p1 is not None and p2 is not None:
                            _draw_thick_line(shader, p1, p2, 3.0, (1.0, 0.2, 0.15, 0.65))
                            _draw_square(shader, p1, 5.0, (1.0, 0.2, 0.15, 0.95))
                            _draw_square(shader, p2, 5.0, (1.0, 0.2, 0.15, 0.95))

                # 十字准星
                coords = [(x - 10, y), (x + 10, y), (x, y - 10), (x, y + 10)]
                batch = batch_for_shader(shader, 'LINES', {"pos": coords})
                shader.bind()
                shader.uniform_float("color", (1.0, 0.9, 0.2, 1.0))
                batch.draw(shader)
        except Exception:
            pass  # GPU 绘制失败不中断，名字文字仍会显示

        # 悬停骨骼名（白色文字，便于在各种背景上辨认）——不依赖 GPU 着色器，始终绘制
        if self._hover:
            try:
                font_id = 0
                blf.position(font_id, x + 14, y + 14, 0)
                blf.size(font_id, 14)
                blf.color(font_id, 1.0, 1.0, 1.0, 1.0)
                blf.draw(font_id, self._hover)
            except Exception:
                pass


class BONE_DT_ClearDampedTrackConstraints(bpy.types.Operator):
    """清除选中骨骼及其子链上由本插件创建的阻尼追踪约束（不误删手调约束）"""

    bl_idname = "bone_dt.clear_damped_track_constraints"
    bl_label = "Clear Damped Track Constraints in Selected Chain"
    bl_description = "Remove the add-on's Damped Track constraints from the selected bones and their descendants"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == 'ARMATURE'

    def execute(self, context):
        arm_obj = context.active_object
        selected_names = get_selected_bone_names(context)
        if not selected_names:
            self.report({'ERROR'}, "Select bones first (Pose or Edit Mode)")
            return {'CANCELLED'}

        target_names = get_bone_and_descendants(arm_obj, selected_names)
        current_mode = context.mode
        try:
            if current_mode != 'POSE':
                bpy.ops.object.mode_set(mode='POSE')

            pose_bones = arm_obj.pose.bones
            removed = 0
            for name in target_names:
                pose_bone = pose_bones.get(name)
                if not pose_bone:
                    continue
                for c in list(pose_bone.constraints):
                    if is_plugin_constraint(c):
                        pose_bone.constraints.remove(c)
                        removed += 1
        finally:
            if current_mode != 'POSE':
                bpy.ops.object.mode_set(mode=current_mode)

        self.report({'INFO'}, f"Cleared {removed} constraints")
        return {'FINISHED'}


class BONE_DT_PT_Panel(bpy.types.Panel):
    bl_label = "Bone Damped Track"
    bl_idname = "BONE_DT_PT_Panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Bone Tools"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        obj = context.active_object

        if not (obj and obj.type == 'ARMATURE'):
            layout.label(text="Select an armature", icon='ERROR')
            return

        box = layout.box()
        box.label(text=f"{obj.name}  |  Bones: {len(obj.data.bones)}")

        layout.separator()
        layout.prop(scene, "bone_dt_scope", expand=True)
        # 仅"全部骨骼"模式提供过滤选项
        if scene.bone_dt_scope == 'ALL':
            layout.prop(scene, "bone_dt_filter_mode", expand=True)
            if scene.bone_dt_filter_mode == 'KEYWORD':
                layout.prop(scene, "bone_dt_keywords", text="Keywords")
                layout.label(text="Bone name / MMD JP-EN names containing any keyword match; comma-separated", icon='INFO')
            elif scene.bone_dt_filter_mode == 'GREEN':
                layout.label(text="Green = has constraints (non-IK); constrained leg bones also match", icon='INFO')
            if scene.bone_dt_filter_mode != 'NONE':
                row = layout.row(align=True)
                row.prop(scene, "bone_dt_manual_include", text="Manual include")
                op = row.operator("bone_dt.pick_bone", text="", icon='EYEDROPPER')
                op.target_prop = "bone_dt_manual_include"
                row = layout.row(align=True)
                row.prop(scene, "bone_dt_manual_exclude", text="Manual exclude")
                op = row.operator("bone_dt.pick_bone", text="", icon='EYEDROPPER')
                op.target_prop = "bone_dt_manual_exclude"
        layout.prop(scene, "bone_dt_influence", text="Start influence")
        layout.prop(scene, "bone_dt_skip_root", text="Skip root bones")
        layout.prop(scene, "bone_dt_child_mode", text="Target child")
        layout.prop(scene, "bone_dt_target_space", text="Target space")
        layout.prop(scene, "bone_dt_overwrite", text="Update existing add-on constraints")

        layout.separator()
        box = layout.box()
        box.label(text="Gradient Mode")
        box.prop(scene, "bone_dt_gradient_mode", expand=True)
        if scene.bone_dt_gradient_mode in ('INCREASE', 'DECREASE'):
            box.prop(scene, "bone_dt_gradient_step", text="Step")
        elif scene.bone_dt_gradient_mode == 'PROPORTIONAL':
            box.prop(scene, "bone_dt_gradient_end", text="End influence")
            box.prop(scene, "bone_dt_gradient_curve", text="Easing")

        layout.separator()
        layout.operator("bone_dt.preview_matched_bones", text="Preview Matched Bones", icon='RESTRICT_SELECT_OFF')
        layout.operator("bone_dt.add_damped_track_to_bones", text="▶ Batch Add", icon='CONSTRAINT')
        layout.operator("bone_dt.clear_damped_track_constraints", text="✖ Clear Selected Chain", icon='X')


classes = [
    BONE_DT_AddDampedTrackToBones,
    BONE_DT_PreviewMatchedBones,
    BONE_DT_PickBoneOperator,
    BONE_DT_ClearDampedTrackConstraints,
    BONE_DT_PT_Panel,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    scene = bpy.types.Scene
    # 带 hasattr 保护，避免脚本热重载（F8）时重复定义属性报错
    if not hasattr(scene, "bone_dt_influence"):
        scene.bone_dt_influence = bpy.props.FloatProperty(
            name="Start Influence", default=1.0, min=0.0, max=1.0, step=1, precision=3,
            description="Base influence value (0-1) for each bone's Damped Track constraint. Higher = bones follow their child tightly (less swing); lower = looser (more swing). Combined with gradient modes for root-stiff / tip-soft results.")
    if not hasattr(scene, "bone_dt_skip_root"):
        scene.bone_dt_skip_root = bpy.props.BoolProperty(
            name="Skip Root Bones", default=True,
            description="When enabled, root bones without parents (anchor bones such as the head or torso) get no constraint; only their descendant chains are processed.")
    if not hasattr(scene, "bone_dt_scope"):
        scene.bone_dt_scope = bpy.props.EnumProperty(
            name="Scope",
            items=[
                ('ALL', "All Bones", "Add constraints to all matching bones in the armature"),
                ('SELECTED', "Selected Chain Only", "Only add constraints to selected bones and their descendants (select in Pose/Edit Mode)"),
            ],
            default='SELECTED',
        )
    if not hasattr(scene, "bone_dt_gradient_mode"):
        scene.bone_dt_gradient_mode = bpy.props.EnumProperty(
            name="Gradient Mode",
            items=[
                ('NONE', "None", "Use the same influence value for all bones"),
                ('INCREASE', "Increase", "Influence += step for each level down the chain"),
                ('DECREASE', "Decrease", "Influence -= step for each level down the chain"),
                ('PROPORTIONAL', "Proportional", "Distribute influence by each chain's total length, from start to end influence, regardless of chain length"),
            ],
            default='NONE',
        )
    if not hasattr(scene, "bone_dt_gradient_end"):
        scene.bone_dt_gradient_end = bpy.props.FloatProperty(
            name="End Influence", default=0.0, min=0.0, max=1.0, precision=3,
            description="Influence at the chain tip in Proportional mode (default 0; raise it to keep tips from going fully loose)",
        )
    if not hasattr(scene, "bone_dt_gradient_curve"):
        scene.bone_dt_gradient_curve = bpy.props.EnumProperty(
            name="Easing",
            items=[
                ('LINEAR', "Linear", "Constant change from start to end value"),
                ('EASE_IN', "Ease In", "Slow near the root, faster toward the tip"),
                ('EASE_OUT', "Ease Out", "Fast near the root, slower toward the tip"),
                ('SMOOTH', "Smooth", "S-shaped transition, gentle at both ends"),
            ],
            default='LINEAR',
        )
    if not hasattr(scene, "bone_dt_gradient_step"):
        scene.bone_dt_gradient_step = bpy.props.FloatProperty(
            name="Step", default=0.1, min=0.0, max=1.0, step=0.1, precision=3,
            description="Influence change per level down the chain in Increase/Decrease mode. Larger step = faster gradient; for long chains prefer Proportional mode.")
    if not hasattr(scene, "bone_dt_child_mode"):
        scene.bone_dt_child_mode = bpy.props.EnumProperty(
            name="Target Child",
            items=[
                ('FIRST', "First Child", "Target the first child bone (default)"),
                ('LAST', "Last Child", "Target the last child bone, for chains needing special control"),
            ],
            default='FIRST',
        )
    if not hasattr(scene, "bone_dt_target_space"):
        scene.bone_dt_target_space = bpy.props.EnumProperty(
            name="Target Space",
            items=[
                ('WORLD', "World", "Solve in world space (default, matches the original)"),
                ('LOCAL', "Local", "Solve in local space; use when the armature's own rotation/scale causes offset"),
            ],
            default='WORLD',
        )
    if not hasattr(scene, "bone_dt_overwrite"):
        scene.bone_dt_overwrite = bpy.props.BoolProperty(
            name="Update Existing Add-on Constraints", default=True,
            description="When re-running Batch Add, update constraints the add-on already created to the current parameters. On = full refresh; off = keep existing constraints, only add to bones without them.")
    if not hasattr(scene, "bone_dt_filter_mode"):
        scene.bone_dt_filter_mode = bpy.props.EnumProperty(
            name="Filter",
            items=[
                ('NONE', "None", "Process all bones"),
                ('GREEN', "Pose-Green Bones", "Only bones that already have constraints (non-IK, with valid targets)"),
                ('KEYWORD', "Cloth/Hair Keywords", "Only bones whose names contain the given cloth/hair keywords"),
            ],
            default='NONE',
        )
    if not hasattr(scene, "bone_dt_keywords"):
        scene.bone_dt_keywords = bpy.props.StringProperty(
            name="Keywords",
            default="髪,髮,hair,スカート,skirt,マント,cape,リボン,ribbon,布,cloth,ポニテ,三つ編み",
            description="Comma-separated; a bone matches if its name, MMD Japanese name or MMD English name contains any keyword",
        )
    if not hasattr(scene, "bone_dt_manual_include"):
        scene.bone_dt_manual_include = bpy.props.StringProperty(
            name="Manual Include Bones",
            default="",
            description="Comma-separated name keywords; bones whose names contain any of them are added even if the filter missed them. Use Preview to verify.",
        )
    if not hasattr(scene, "bone_dt_manual_exclude"):
        scene.bone_dt_manual_exclude = bpy.props.StringProperty(
            name="Manual Exclude Bones",
            default="",
            description="Comma-separated name keywords; bones whose names contain any of them are removed even if the filter matched them (e.g. 足,腿,foot,leg). Use Preview to verify.",
        )


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    props = (
        'bone_dt_influence', 'bone_dt_skip_root', 'bone_dt_scope',
        'bone_dt_gradient_mode', 'bone_dt_gradient_step',
        'bone_dt_gradient_end', 'bone_dt_gradient_curve',
        'bone_dt_child_mode', 'bone_dt_target_space', 'bone_dt_overwrite',
        'bone_dt_filter_mode', 'bone_dt_keywords',
        'bone_dt_manual_include', 'bone_dt_manual_exclude',
        'bone_dt_only_green', 'bone_dt_only_deform',  # 清理旧版本遗留的废弃属性
    )
    for p in props:
        if hasattr(bpy.types.Scene, p):
            delattr(bpy.types.Scene, p)


if __name__ == "__main__":
    register()
