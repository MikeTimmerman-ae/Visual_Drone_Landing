import pybullet as p


def create_pad():
    # Create plane for ground
    p.createCollisionShape(p.GEOM_PLANE)
    p.createMultiBody(0, 0)

    # Parameters for the pad (base and cross)
    pad_size = 500.0
    cross_size = 1.
    pad_thickness = 0.01

    # Create the base collision and visual shapes for the pad
    padColBoxId = p.createCollisionShape(p.GEOM_BOX, halfExtents=[pad_size / 2, pad_size / 2, pad_thickness])
    padVisualShapeId = p.createVisualShape(p.GEOM_BOX, halfExtents=[pad_size / 2, pad_size / 2, pad_thickness], rgbaColor=[0, 1, 0, 1])  # Green color

    # Create the cross shapes (red) as parts of the multibody
    crossColBoxId = p.createCollisionShape(p.GEOM_BOX, halfExtents=[cross_size / 2, pad_size / 100, 2 * pad_thickness])
    crossVisualShapeId = p.createVisualShape(p.GEOM_BOX, halfExtents=[cross_size / 2, pad_size / 100, 2 * pad_thickness], rgbaColor=[1, 0, 0, 1])  # Red color

    crossColBoxId2 = p.createCollisionShape(p.GEOM_BOX, halfExtents=[pad_size / 100, cross_size / 2, 2 * pad_thickness])
    crossVisualShapeId2 = p.createVisualShape(p.GEOM_BOX, halfExtents=[pad_size / 100, cross_size / 2, 2 * pad_thickness], rgbaColor=[1, 0, 0, 1])  # Red color

    # Create boxes
    boxId1 = p.createCollisionShape(p.GEOM_BOX, halfExtents=[cross_size / 2, cross_size / 2, 2 * pad_thickness])
    boxVisualShapeId1 = p.createVisualShape(p.GEOM_BOX, halfExtents=[cross_size / 2, cross_size / 2, 2 * pad_thickness], rgbaColor=[1, 0, 0, 1])  # Red color

    boxId2 = p.createCollisionShape(p.GEOM_BOX, halfExtents=[cross_size / 2, cross_size / 2, 2 * pad_thickness])
    boxVisualShapeId2 = p.createVisualShape(p.GEOM_BOX, halfExtents=[cross_size / 2, cross_size / 2, 2 * pad_thickness],
                                            rgbaColor=[1, 0, 0, 1])  # Red color

    boxId3 = p.createCollisionShape(p.GEOM_BOX, halfExtents=[cross_size / 2, cross_size / 2, 2 * pad_thickness])
    boxVisualShapeId3 = p.createVisualShape(p.GEOM_BOX, halfExtents=[cross_size / 2, cross_size / 2, 2 * pad_thickness],
                                            rgbaColor=[1, 0, 0, 1])  # Red color

    boxId4 = p.createCollisionShape(p.GEOM_BOX, halfExtents=[cross_size / 2, cross_size / 2, 2 * pad_thickness])
    boxVisualShapeId4 = p.createVisualShape(p.GEOM_BOX, halfExtents=[cross_size / 2, cross_size / 2, 2 * pad_thickness],
                                            rgbaColor=[1, 0, 0, 1])  # Red color

    # Link parameters
    link_Masses = [1, 1, 1, 1, 1, 1, 1]  # The mass for the pad and cross pieces
    linkCollisionShapeIndices = [padColBoxId, crossColBoxId, crossColBoxId2, boxId1, boxId2, boxId3, boxId4]  # Collision shapes
    linkVisualShapeIndices = [padVisualShapeId, crossVisualShapeId, crossVisualShapeId2, boxVisualShapeId1, boxVisualShapeId2, boxVisualShapeId3, boxVisualShapeId4]  # Visual shapes
    linkPositions = [[0, 0, pad_thickness],
                     [0, 0, 5*pad_thickness],
                     [0, 0, 5*pad_thickness],
                     [pad_size / 100, pad_size / 100, 5*pad_thickness],
                     [pad_size / 100, -pad_size / 100, 5*pad_thickness],
                     [-pad_size / 100, pad_size / 100, 5*pad_thickness],
                     [-pad_size / 100, -pad_size / 100, 5*pad_thickness]]  # Positioning of the parts
    linkOrientations = [[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]]  # Orientations (no rotation)
    linkInertialFramePositions = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]  # Inertial frame positions
    linkInertialFrameOrientations = [[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]]  # Inertial frame orientations
    indices = [0, 0, 0, 0, 0, 0, 0]  # Parent indices (all linked to the base)
    jointTypes = [p.JOINT_FIXED, p.JOINT_FIXED, p.JOINT_FIXED, p.JOINT_FIXED, p.JOINT_FIXED, p.JOINT_FIXED, p.JOINT_FIXED]  # Fixed joints (no movement between parts)
    axis = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]  # No axis for fixed joints

    # Base position and orientation for the pad
    basePosition = [0, 0, pad_thickness]  # Base of the landing pad
    baseOrientation = [0, 0, 0, 1]  # No rotation

    # Create the multi-body system (pad and cross)
    pad_id = p.createMultiBody(
        baseMass=0,  # Mass of the base (no mass for the plane)
        baseCollisionShapeIndex=padColBoxId,
        baseVisualShapeIndex=padVisualShapeId,
        basePosition=basePosition,
        baseOrientation=baseOrientation,
        linkMasses=link_Masses,
        linkCollisionShapeIndices=linkCollisionShapeIndices,
        linkVisualShapeIndices=linkVisualShapeIndices,
        linkPositions=linkPositions,
        linkOrientations=linkOrientations,
        linkInertialFramePositions=linkInertialFramePositions,
        linkInertialFrameOrientations=linkInertialFrameOrientations,
        linkParentIndices=indices,
        linkJointTypes=jointTypes,
        linkJointAxis=axis
    )
    return pad_id
