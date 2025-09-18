import logging
import os
import math
import vtk
import SimpleITK as sitk
import numpy as np
import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
# from myDosenet import myDosenet
from plans.brachy_plan import brachy_plan
from plans.config import setting
from plans.utilizations import position_transform, direction_transform
from MarkupConstraints import MarkupConstraintsLogic, ControlPoint
# import torch
import copy
import ast
import sitkUtils
logic = MarkupConstraintsLogic()

#
# AddSources
#

class AddSources(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "AddSources"  # TODO: make this more human readable by adding spaces
        self.parent.categories = ["BrachyPlan"]  # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = [""]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        self.parent.helpText = """
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#AddSources">module documentation</a>.
"""
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
"""

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#

def registerSampleData():
    """
    Add data sets to Sample Data module.
    """
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData
    iconsPath = os.path.join(os.path.dirname(__file__), 'Resources/Icons')

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # AddSources1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category='AddSources',
        sampleName='AddSources1',
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, 'AddSources1.png'),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames='AddSources1.nrrd',
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums='SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95',
        # This node name will be used when the data set is loaded
        nodeNames='AddSources1'
    )

    # AddSources2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category='AddSources',
        sampleName='AddSources2',
        thumbnailFileName=os.path.join(iconsPath, 'AddSources2.png'),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames='AddSources2.nrrd',
        checksums='SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97',
        # This node name will be used when the data set is loaded
        nodeNames='AddSources2'
    )


#
# AddSourcesWidget
#

class AddSourcesWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._updatingGUIFromParameterNode = False

    def setup(self):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/AddSources.ui'))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = AddSourcesLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # These connections ensure that whenever user changes some settings on the GUI, that is saved in the MRML scene
        # (in the selected parameter node).
        # 信号槽，编辑
        self.ui.ctvSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.ctvSelector.setMRMLScene(slicer.mrmlScene)
        self.ui.inputSelector2.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.planeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        # self.ui.imageThresholdSliderWidget.connect("valueChanged(double)", self.updateParameterNodeFromGUI)
        # self.ui.invertOutputCheckBox.connect("toggled(bool)", self.updateParameterNodeFromGUI)
        # self.ui.markupSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.lineSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.lineSelector.setMRMLScene(slicer.mrmlScene)

        self.ui.lineSelector_2.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.lineSelector_2.setMRMLScene(slicer.mrmlScene)

        self.ui.markupSelector_2.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.markupSelector_2.setMRMLScene(slicer.mrmlScene)

        # Buttons
        self.ui.applyButton.connect('clicked(bool)', self.onApplyButton)
        self.ui.addNeedleButton.connect('clicked(bool)', self.addLineButton)
        self.ui.setSeedButton.connect('clicked(bool)', self.setSeedButton)
        self.ui.planButton.connect('clicked(bool)', self.planButton)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

    def cleanup(self):
        """
        Called when the application closes and the module widget is destroyed.
        """
        self.removeObservers()

    def enter(self):
        """
        Called each time the user opens this module.
        """
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self):
        """
        Called each time the user opens a different module.
        """
        # Do not react to parameter node changes (GUI wlil be updated when the user enters into the module)
        self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

    def onSceneStartClose(self, caller, event):
        """
        Called just before the scene is closed.
        """
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event):
        """
        Called just after the scene is closed.
        """
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self):
        """
        Ensure parameter node exists and observed.
        """
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        # "InputVolume"---markups "InputVolume2"---vtkMRMLScalarVolumeNode
        if not self._parameterNode.GetNodeReference("MarkupVolume"):
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLMarkupsFiducialNode")
            if firstVolumeNode:
                self._parameterNode.SetNodeReferenceID("MarkupVolume", firstVolumeNode.GetID())
        
        if not self._parameterNode.GetNodeReference("MarkupVolume2"):
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLMarkupsFiducialNode")
            if firstVolumeNode:
                self._parameterNode.SetNodeReferenceID("MarkupVolume2", firstVolumeNode.GetID())
        
        if not self._parameterNode.GetNodeReference("InputVolume2"):
            firstVolumeNode2 = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode2:
                self._parameterNode.SetNodeReferenceID("InputVolume2", firstVolumeNode2.GetID())

        if not self._parameterNode.GetNodeReference("ctvVolume"):
            firstVolumeNode6 = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLLabelMapVolumeNode")
            if firstVolumeNode6:
                self._parameterNode.SetNodeReferenceID("ctvVolume", firstVolumeNode6.GetID())

        if not self._parameterNode.GetNodeReference("PlaneVolume"):
            firstVolumeNode3 = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLMarkupsPlaneNode")
            if firstVolumeNode3:
                self._parameterNode.SetNodeReferenceID("PlaneVolume", firstVolumeNode3.GetID())

        if not self._parameterNode.GetNodeReference("LineVolume"):
            firstVolumeNode4 = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLMarkupsLineNode")
            if firstVolumeNode4:
                self._parameterNode.SetNodeReferenceID("LineVolume", firstVolumeNode4.GetID())

        if not self._parameterNode.GetNodeReference("LineVolume2"):
            firstVolumeNode4 = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLMarkupsLineNode")
            if firstVolumeNode4:
                self._parameterNode.SetNodeReferenceID("LineVolume2", firstVolumeNode4.GetID())

        if not self._parameterNode.GetNodeReference("SegmentVolume"):
            firstVolumeNode5 = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLSegmentationNode")
            if firstVolumeNode5:
                self._parameterNode.SetNodeReferenceID("SegmentVolume", firstVolumeNode5.GetID())



    def setParameterNode(self, inputParameterNode): # 不动
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if inputParameterNode:
            self.logic.setDefaultParameters(inputParameterNode)

        # Unobserve previously selected parameter node and add an observer to the newly selected.
        # Changes of parameter node are observed so that whenever parameters are changed by a script or any other module
        # those are reflected immediately in the GUI.
        if self._parameterNode is not None and self.hasObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode):
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
        self._parameterNode = inputParameterNode
        if self._parameterNode is not None:
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

        # Initial GUI update
        self.updateGUIFromParameterNode()


    def updateGUIFromParameterNode(self, caller=None, event=None):
        """
        This method is called whenever parameter node is changed.
        The module GUI is updated to show the current state of the parameter node.
        """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        # Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
        self._updatingGUIFromParameterNode = True

        # Update node selectors and sliders
        self.ui.ctvSelector.setCurrentNode(self._parameterNode.GetNodeReference("ctvVolume"))
        self.ui.inputSelector2.setCurrentNode(self._parameterNode.GetNodeReference("InputVolume2"))
        self.ui.planeSelector.setCurrentNode(self._parameterNode.GetNodeReference("PlaneVolume"))
        self.ui.markupSelector.setCurrentNode(self._parameterNode.GetNodeReference("MarkupVolume"))
        self.ui.lineSelector.setCurrentNode(self._parameterNode.GetNodeReference("LineVolume"))
        self.ui.markupSelector_2.setCurrentNode(self._parameterNode.GetNodeReference("MarkupVolume2"))
        self.ui.lineSelector_2.setCurrentNode(self._parameterNode.GetNodeReference("LineVolume2"))
        self.ui.segmentSelector.setCurrentNode(self._parameterNode.GetNodeReference("SegmentVolume"))
        # self.ui.imageThresholdSliderWidget.value = float(self._parameterNode.GetParameter("Threshold"))
        # self.ui.invertOutputCheckBox.checked = (self._parameterNode.GetParameter("Invert") == "true")

        # Update buttons states and tooltips
        if (self._parameterNode.GetNodeReference("MarkupVolume") and self._parameterNode.GetNodeReference("InputVolume2") and self._parameterNode.GetNodeReference("SegmentVolume")):
            self.ui.applyButton.toolTip = "Compute output volume"
            self.ui.applyButton.enabled = True
        else:
            self.ui.applyButton.toolTip = "Select input volume/markups and output volume nodes"
            self.ui.applyButton.enabled = False

        # All the GUI updates are done
        self._updatingGUIFromParameterNode = False

    def updateParameterNodeFromGUI(self, caller=None, event=None):
        """
        This method is called when the user makes any change in the GUI.
        The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
        """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        wasModified = self._parameterNode.StartModify()  # Modify all properties in a single batch

        self._parameterNode.SetNodeReferenceID("ctvVolume", self.ui.ctvSelector.currentNodeID)
        self._parameterNode.SetNodeReferenceID("InputVolume2", self.ui.inputSelector2.currentNodeID)
        self._parameterNode.SetNodeReferenceID("PlaneVolume", self.ui.planeSelector.currentNodeID)
        # self._parameterNode.SetParameter("Threshold", str(self.ui.imageThresholdSliderWidget.value))
        # self._parameterNode.SetParameter("Invert", "true" if self.ui.invertOutputCheckBox.checked else "false")
        # self._parameterNode.SetNodeReferenceID("MarkupVolume", self.ui.markupSelector.currentNodeID)
        self._parameterNode.SetNodeReferenceID("LineVolume", self.ui.lineSelector.currentNodeID)
        self._parameterNode.SetNodeReferenceID("MarkupVolume2", self.ui.markupSelector_2.currentNodeID)
        self._parameterNode.SetNodeReferenceID("LineVolume2", self.ui.lineSelector_2.currentNodeID)
        # self._parameterNode.SetNodeReferenceID("SegmentVolume", self.ui.segmentSelector.currentNodeID)

        self._parameterNode.EndModify(wasModified)

    def onApplyButton(self):
        """
        Run processing when user clicks "Apply" button.
        """
        with slicer.util.tryWithErrorDisplay("Failed to compute results.", waitCursor=True):

            # Compute output
            self.logic.process(self.ui.inputSelector2.currentNode())


    def addLineButton(self):
        with slicer.util.tryWithErrorDisplay("Failed to compute results.", waitCursor=True):
            self.logic.addLine(self.ui.planeSelector.currentNode(), self.ui.lineSelector.currentNode(), self.ui.inputSelector2.currentNode())

    def setSeedButton(self):
        with slicer.util.tryWithErrorDisplay("Failed to compute results.", waitCursor=True):
            self.logic.setSeed(self.ui.lineSelector_2.currentNode(), self.ui.markupSelector_2.currentNode(), self.ui.inputSelector2.currentNode())

    def planButton(self):
        with slicer.util.tryWithErrorDisplay("Failed to compute results.", waitCursor=True):
            self.logic.planButton(self.ui.inputSelector2.currentNode(), self.ui.ctvSelector.currentNode())
#
# AddSourcesLogic
#

class AddSourcesLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self):
        """
        Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        ScriptedLoadableModuleLogic.__init__(self)

    # def setDefaultParameters(self, parameterNode):
        """
        Initialize parameter node with default settings.
        """
        # if not parameterNode.GetParameter("Threshold"):
        #     parameterNode.SetParameter("Threshold", "100.0")
        # if not parameterNode.GetParameter("Invert"):
        #     parameterNode.SetParameter("Invert", "false")
    def planButton(self, inputVolume2, inputctv):
        import time
        startTime = time.time()
        logging.info('Processing started')
         # 获取与输入 Volume 关联的 Subject Hierarchy 节点
        shNode = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
        volume1ItemID = shNode.GetItemByDataNode(inputVolume2)

        # 查找输入 Volume 的 Study 级别
        studyItemID = shNode.GetItemAncestorAtLevel(volume1ItemID, 'Study')
        # print(studyItemID)
        # ctimage几何信息
        ctimage = self.get_sitk_image(inputVolume2)
        ctimage_vtk = inputVolume2.GetImageData()
        dimension = ctimage_vtk.GetDimensions()
        spacing = inputVolume2.GetSpacing()
        origin = inputVolume2.GetOrigin()
        fMat = vtk.vtkMatrix4x4()
        inputVolume2.GetIJKToRASDirectionMatrix(fMat)
        segmentation_image = self.get_sitk_image(inputctv)

        args = setting()
        plan_res, sum_image, dose_image = brachy_plan(ctimage, segmentation_image, args)
        
        planned_seeds = []
        planned_seed_doses = []
        for res in plan_res:
            planned_seeds.append(res[1])
            print("seeds", res[1])
            planned_seed_doses.append(res[2])
        for i, seeds in enumerate(planned_seeds):
            lineNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsLineNode')
            lineID = shNode.GetItemByDataNode(lineNode)
            shNode.SetItemParent(lineID, volume1ItemID)
            fiducialNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode')
            fiducialNode.SetName(f'{inputctv.GetName()}_seed_{i}')
            pointsID = shNode.GetItemByDataNode(fiducialNode)
            shNode.SetItemParent(pointsID, lineID)
            for j, seed in enumerate(seeds):
                print('seed_num',i,j)
            # Transform the seed position into physical space
                pos = position_transform(dose_image, seed[0])
                fiducialNode.AddControlPoint(pos)
                print('pos',pos)
                # Transform and normalize the seed direction
                direction = direction_transform(dose_image, seed[1])
                print('direction',direction)

                if j ==0:
                    lineNode.AddControlPoint(vtk.vtkVector3d(pos[0],pos[1],pos[2]))  
                if j == len(seeds)-1:
                    lineNode.AddControlPoint(vtk.vtkVector3d(pos[0],pos[1],pos[2]))

                
                self.create_capsule_stl(j, fiducialNode, pos, direction)
            lineNode.SetIJKToRASDirectionMatrix(fMat)
            fiducialNode.SetIJKToRASDirectionMatrix(fMat)
        
        # total_dose = np.sum(np.array(planned_seed_doses), axis=0)

        # ##直接生成doseVolume的节点
        dose_ = sitk.GetImageFromArray(sum_image)
        dose_.SetSpacing(spacing)
        dose_ = self.ImageResample_size(dose_, dimension)
        # dose_ = self.ImageResample_spacing(dose_, spacing)
        dose_node = slicer.util.addVolumeFromArray(sitk.GetArrayFromImage(dose_), nodeClassName="vtkMRMLScalarVolumeNode") 
        doseimage = dose_node.GetImageData()
        dosedimension = doseimage.GetDimensions()

        # print('dosedimension',dosedimension)
        # 设置属性
        dose_node.SetSpacing(spacing)
        dose_node.SetOrigin(origin)
        dose_node.SetIJKToRASDirectionMatrix(fMat)

        doseID = shNode.GetItemByDataNode(dose_node)
        shNode.SetItemParent(doseID, volume1ItemID)
        shNode.SetItemName(doseID, 'RTDoseMap')
        # # 将doseimage加入study下
        if studyItemID:
            shNode.SetItemParent(doseID, studyItemID)

         #  dose name value 在 slicer 中的属性名称, 确认所属的study 是否已经包含了这个变量
        doseUnitNameInStudy = shNode.GetItemAttribute(studyItemID,"DicomRtImport.DoseUnitName")
        doseUnitValueInStudy = shNode.GetItemAttribute(studyItemID,"DicomRtImport.DoseUnitValue")

        defaultDoseUnitName = doseUnitNameInStudy if doseUnitNameInStudy else "Gy"
        defaultDoseUnitValue = doseUnitValueInStudy if doseUnitValueInStudy else "1.0"
        # 如果study中没有dose name value

        shNode.SetItemAttribute(studyItemID, "DicomRtImport.DoseUnitName", defaultDoseUnitName)
        shNode.SetItemAttribute(studyItemID, "DicomRtImport.DoseUnitValue", defaultDoseUnitValue)

        dose_node.SetAttribute("DicomRtImport.DoseVolume", "1")

        # 请求所有者插件搜索
        shNode.RequestOwnerPluginSearch(doseID)

        # 通知项已被修改
        shNode.ItemModified(doseID)

        stopTime = time.time()
        logging.info(f'Processing completed in {stopTime-startTime:.2f} seconds')
        print(f'Processing completed in {stopTime-startTime:.2f} seconds')

    def get_sitk_image(self,mrml_volume_node):
         # 从 MRML 节点获取 numpy 数组
        array = slicer.util.arrayFromVolume(mrml_volume_node)
    
         # 获取图像的方向、间距和起始位置
        spacing = mrml_volume_node.GetSpacing()
        origin = mrml_volume_node.GetOrigin()
        fMat = vtk.vtkMatrix4x4()
        mrml_volume_node.GetIJKToRASDirectionMatrix(fMat)
        direction = np.array(fMat).flatten()[:9].tolist()
        # 将 numpy 数组转换为 SimpleITK Image
        sitk_image = sitk.GetImageFromArray(array)
        sitk_image.SetSpacing(spacing)
        sitk_image.SetOrigin(origin)
        # sitk_image.SetDirection(direction)
        
        return sitk_image

            
    def addLine(self, inputVolume, inputLine, inputCT):
    # 将线段调整为垂直参考面的方向
        if not inputVolume or not inputLine:
            raise ValueError("Input plane or line is invalid")
        
        normal = np.empty(3)
        inputVolume.GetNormal(normal)
        point1 = np.empty(3)
        inputLine.GetPosition1(point1)
        point2 = np.empty(3)
        inputLine.GetPosition2(point2)
        distance = math.sqrt(sum((point1[i] - point2[i])**2 for i in range(3)))
        # print(distance)

        normal_unit = np.array(normal) / np.linalg.norm(normal)
        # print(normal_unit)
        move_vector = normal_unit * distance
        # print(move_vector)

        # 将端点2沿着法向量移动
        new_point2 = [point1[i] + move_vector[i] for i in range(3)]

        # 更新线段的端点
        inputLine.SetPosition2(new_point2)
        
        shNode = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
        planeID = shNode.GetItemByDataNode(inputVolume)
        lineID = shNode.GetItemByDataNode(inputLine)
        ctID = shNode.GetItemByDataNode(inputCT)
        shNode.SetItemParent(planeID, ctID)
        shNode.SetItemParent(lineID, ctID)

        

    def create_capsule_stl(self, seed_num, input_points_node, center, direction, length = 3.7, radius = 0.4, resolution=60, name = None):

        # print('center',center)
        magnitude = math.sqrt(sum([x**2 for x in direction]))  # Calculate the magnitude
        direction = [x / magnitude for x in direction] 
        # center = [-center[0],-center[1],center[2]]

        start_point = (center[0] - 0.5 * length * direction[0],
                    center[1] - 0.5 * length * direction[1],
                    center[2] - 0.5 * length * direction[2])

        end_point = (center[0] + 0.5 * length * direction[0],
                    center[1] + 0.5 * length * direction[1],
                    center[2] + 0.5 * length * direction[2])

        # 创建一个圆柱体
        cylinder = vtk.vtkCylinderSource()
        cylinder.SetRadius(radius)
        cylinder.SetHeight(length)
        cylinder.SetResolution(100)

        # 计算旋转矩阵
        length = np.linalg.norm(direction)
        target_direction = direction / length
        current_direction = np.array([0.0, 1.0, 0.0])  # 初始方向，这里假设圆柱体的初始方向是Y轴
        rotation_axis = np.cross(current_direction, target_direction)
        rotation_angle = np.arccos(np.dot(current_direction, target_direction) / (np.linalg.norm(current_direction) * np.linalg.norm(target_direction)))
        
        # 旋转圆柱体
        transform = vtk.vtkTransform()
        transform.RotateWXYZ(np.degrees(rotation_angle), rotation_axis)
        transform_filter = vtk.vtkTransformPolyDataFilter()
        transform_filter.SetInputConnection(cylinder.GetOutputPort())
        transform_filter.SetTransform(transform)
        transform_filter.Update()

        # 平移圆柱体
        transform2 = vtk.vtkTransform()
        transform2.Translate(center)
        transform_filter2 = vtk.vtkTransformPolyDataFilter()
        transform_filter2.SetInputConnection(transform_filter.GetOutputPort())
        transform_filter2.SetTransform(transform2)
        transform_filter2.Update()

        tri1 = vtk.vtkTriangleFilter()
        tri1.SetInputConnection(transform_filter2.GetOutputPort())
        tri1.Update()

        # Create two hemispheres for the ends of the capsule
        sphere1 = vtk.vtkSphereSource()
        sphere1.SetRadius(radius+0.01)
        sphere1.SetPhiResolution(resolution)
        sphere1.SetThetaResolution(resolution)
        sphere1.SetCenter(start_point)

        tri2 = vtk.vtkTriangleFilter()
        tri2.SetInputConnection(sphere1.GetOutputPort())
        tri2.Update()

        sphere2 = vtk.vtkSphereSource()
        sphere2.SetRadius(radius+0.01)
        sphere2.SetPhiResolution(resolution)
        sphere2.SetThetaResolution(resolution)
        sphere2.SetCenter(end_point)

        tri3 = vtk.vtkTriangleFilter()
        tri3.SetInputConnection(sphere2.GetOutputPort())
        tri3.Update()

        appendFilter = vtk.vtkAppendPolyData()
        appendFilter.AddInputData(tri1.GetOutput())
        appendFilter.AddInputData(tri2.GetOutput())
        appendFilter.AddInputData(tri3.GetOutput())
        appendFilter.Update()
        triangleFilter = vtk.vtkTriangleFilter()
        triangleFilter.SetInputConnection(appendFilter.GetOutputPort())
        triangleFilter.Update()
        
        shNode = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
        pointsID = shNode.GetItemByDataNode(input_points_node)
        # # 删除旧的粒子模型
        # old_seed_ids = []
        # shNode.GetItemChildren(pointsID, old_seed_ids)
        # shNode.RemoveItemChildren(pointsID)

        # 添加新的粒子节点 类型model
        seed_node = slicer.modules.models.logic().AddModel(triangleFilter.GetOutput())
        seed_node.SetName(f'{input_points_node.GetName()}_seed_{seed_num}')
        # seed_node.SetPolyDataConnection(triangleFilter.GetOutputPort())
        seedID = shNode.GetItemByDataNode(seed_node)

        descriptionText = np.array2string(np.array(direction))
        shNode.SetItemAttribute(seedID, "Direction", descriptionText)
        descriptionText = np.array2string(np.array(center))
        shNode.SetItemAttribute(seedID, "Center", descriptionText)
        shNode.SetItemParent(seedID, pointsID)
        # 请求所有者插件搜索
        shNode.RequestOwnerPluginSearch(seedID)
        # 通知项已被修改
        shNode.ItemModified(seedID)
  
        
    def setSeed(self, inputLine, inputPoints, inputCT):
    # 将点的位置移动到线段上
        if not inputPoints or not inputLine:
            raise ValueError("Input line or point list is invalid")
        
        numPoints = inputPoints.GetNumberOfControlPoints()
        # print('num', numPoints)
        point1 = np.empty(3)
        inputLine.GetLineStartPosition(point1)
        point2 = np.empty(3)
        inputLine.GetLineEndPosition(point2)
        line_direction = point2 - point1
        line_length = math.sqrt(sum((point1[i] - point2[i])**2 for i in range(3)))

        pos_n = [0, 0, 0]

        shNode = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
        pointsID = shNode.GetItemByDataNode(inputPoints)
        # 删除旧的粒子模型
        old_seed_ids = []
        shNode.GetItemChildren(pointsID, old_seed_ids)
        shNode.RemoveItemChildren(pointsID)

        for m in range(0, numPoints):
            inputPoints.GetNthControlPointPosition(m, pos_n)
        
            # 计算从线段起点到目标点的向量
            vector_to_target = pos_n - point1

            # 计算点积
            dot_product = np.dot(vector_to_target, line_direction)

            # 计算距离线段起点最近的投影
            projection_point = point1 + (dot_product / line_length**2) * line_direction
            
            inputPoints.SetNthControlPointPosition(m, projection_point)

            self.create_capsule_stl(m, inputPoints, projection_point, line_direction)

        shNode = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
        pointsID = shNode.GetItemByDataNode(inputPoints)
        lineID = shNode.GetItemByDataNode(inputLine)
        ctID = shNode.GetItemByDataNode(inputCT)
        shNode.SetItemParent(pointsID, ctID)
        shNode.SetItemParent(lineID, ctID)
    
    def point_source_map(self, seed, image_origin, image_size, image_spacing):
        x = image_origin[0] + np.arange(image_size[0]) * image_spacing[0]
        y = image_origin[1] + np.arange(image_size[1]) * image_spacing[1]
        z = image_origin[2] + np.arange(image_size[2]) * image_spacing[2]

        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        distance_squared = (X - seed[0])**2 + (Y - seed[1])**2 + (Z - seed[2])**2
        distance = np.sqrt(distance_squared)
        print(np.where(distance_squared<0.01))
        distance[np.where(distance<0.01)] = 0.01
        # with np.errstate(divide='ignore'):  # 忽略除以零的警告
        #     distance_map = np.where(distance != 0, 1.0 / distance, 0.0)
        distance_map = np.transpose(distance, (2, 1, 0))
        print(np.max(distance_map))
        return distance_map
    
    def line_source_map(self, seed, image_origin, image_size, image_spacing):
    
        x = image_origin[0] + np.arange(image_size[0]) * image_spacing[0]
        y = image_origin[1] + np.arange(image_size[1]) * image_spacing[1]
        z = image_origin[2] + np.arange(image_size[2]) * image_spacing[2]

        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        Vx = X - seed[0]
        Vy = Y - seed[1]
        Vz = Z - seed[2]

        distance_squared = Vx**2 + Vy**2 + Vz**2
        # print(np.where(distance_squared<0.01))
        distance_squared[np.where(distance_squared<0.01)] = 0.01
        v0 = np.array([0, 0, 1])
        L = 4.5
        
        
        # # 将夹角从弧度转换为角度
        # # phi_degrees = np.degrees(phi)
        end_1 = np.array([0, 0, L/2])
        end_2 = np.array([0, 0, -L/2])
        # A_prime = Rx @ Ry @ end_1 + seed[0:3]
        # B_prime = Rx @ Ry @ end_2 + seed[0:3]

        norm_direction_vector = seed[3] / np.linalg.norm(seed[3])
        norm_direction_vector = np.array([norm_direction_vector[0], norm_direction_vector[1], -norm_direction_vector[2]])
        # 计算两个点的坐标
        A_prime = seed[0:3] - L/2 * norm_direction_vector
        B_prime = seed[0:3] + L/2 * norm_direction_vector

        # # print('中点', (A_prime+B_prime)*0.5)

        V_PA = np.array([X - A_prime[0], Y - A_prime[1], Z - A_prime[2]])
        V_PB = np.array([X - B_prime[0], Y - B_prime[1], Z - B_prime[2]])

        # 计算向量的长度
        V_PA_magnitude = np.sqrt(np.sum(V_PA**2, axis=0))
        V_PB_magnitude = np.sqrt(np.sum(V_PB**2, axis=0))

        # 计算向量之间的夹角
        dot_product = np.sum(V_PA * V_PB, axis=0)
        cos_beta = np.abs(dot_product / (V_PA_magnitude * V_PB_magnitude))
        beta = np.arccos(np.clip(cos_beta, -1.0, 1.0))

        # # 计算夹角的余弦值
        # cos_beta = dot_product / (norm_a * norm_b)

        # # 计算夹角（以弧度为单位）
        # beta = np.arccos(cos_beta)

        vectors_to_mid_point = np.stack((X - seed[0], Y - seed[1], Z - seed[2]), axis=-1)
        
        norm_vectors_to_mid_point = vectors_to_mid_point / np.linalg.norm(vectors_to_mid_point, axis=-1, keepdims=True)
        cos_theta = np.dot(norm_vectors_to_mid_point, norm_direction_vector)
        theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))

        line_map = beta/ (np.sin(theta) * (distance_squared)**(-1) + 1e-5)

        # 取倒数
        # line_map = (np.sin(theta) * (distance_squared)**(-1) + 1e-5) / beta 
        # print(np.max(line_map))
        # mask = (theta == 0)
        # map_where = (distance_squared - L**2 /4) ** (-1)
        # line_map[mask] = map_where[mask]
        # line_map = np.transpose(line_map, (2, 1, 0))
        line_map = np.transpose(line_map, (2, 1, 0))
        return line_map
    
    def position_soft_method(self, seed, image_origin, image_size, image_spacing):
    
        sphere_radius = 4 
        sphere_volume = (4/3) * np.pi * sphere_radius**3
        
        # 创建一个空的 3D 网格用于软治疗计划
        grid_shape = tuple(image_size)  # 示例网格形状，根据需要调整
        soft_treatment_plan = np.zeros(grid_shape)
        grid_spacing = np.array(image_spacing)

        x_grid, y_grid, z_grid = np.meshgrid(np.arange(grid_shape[0]),
                                                np.arange(grid_shape[1]),
                                                np.arange(grid_shape[2]),
                                                indexing='ij')

        voxel_centers = np.stack([x_grid, y_grid, z_grid], axis=-1) * grid_spacing + image_origin
            
        # 计算所有体素中心到球体中心的距离
        distances = np.linalg.norm(voxel_centers - seed[0:3], axis=-1)
                # 检查体素中心是否在球体内部
        overlap_mask = distances <= sphere_radius  # 只计算球体内的体素
        overlapping_volume = overlap_mask * (4/3) * np.pi * ((sphere_radius - distances) ** 3)
            
        # 归一化重叠体积
        normalized_volume = overlapping_volume / sphere_volume
            
            # 将归一化的重叠体积加到软治疗计划中
        soft_treatment_plan += normalized_volume

        # 归一化软治疗计划，确保重叠体素的总和为 1
        soft_treatment_plan[soft_treatment_plan > 0] /= np.sum(soft_treatment_plan[soft_treatment_plan > 0])
        soft_treatment_plan = np.transpose(soft_treatment_plan, (2, 1, 0))
        return soft_treatment_plan
    
    def subimage_generator(self, image, patch_block_size, stride):
        """
        generate the sub images with patch_block_size, 返回值数组，形状[n, patch_block_size]
        """
        y = np.shape(image)[1]
        x = np.shape(image)[2]
        imagez = np.shape(image)[0]

        blocky = np.array(patch_block_size)[1]
        blockx = np.array(patch_block_size)[0]
        blockz = np.array(patch_block_size)[2]

        stridey = stride[1]
        stridex = stride[0]
        stridez = stride[2]

        if imagez > blockz:
            number_z = (imagez - blockz) // stridez + 1
        else:
            number_z = 1
            # pad_array = np.zeros((blockz, blocky, blockx))
            # pad_array[:imagez, :y, :x] = image
            # image = copy.deepcopy(pad_array)

            # pad_array_label = np.zeros((blockz, blocky, blockx))
            # mask = copy.deepcopy(pad_array_label)
            # pad_array_label[:imagez, :y, :x] = mask

        number_y = (y - blocky) // stridey + 1
        number_x = (x - blockx) // stridex + 1
        hr_samples_list = []

        for i in range(number_z):
            for j in range(number_y):
                for k in range(number_x):
                    # 计算窗口的起始和结束索引
                    start_d = i * stridez
                    end_d = start_d + blockz
                    start_h = j * stridey
                    end_h = start_h + blocky
                    start_w = k * stridex
                    end_w = start_w + blockx

                    # 提取窗口，注意 [z, y, x] 顺序
                    image_block = image[start_d:end_d, start_h:end_h, start_w:end_w]
                    hr_samples_list.append(image_block)
                    
                    
        hr_samples = np.array(hr_samples_list).reshape((len(hr_samples_list), blockz, blocky, blockx))
        return hr_samples
    
    def restore_image(self, original_shape, hr_samples, patch_block_size, stride):
         
        imagez, y, x = original_shape
        blocky = np.array(patch_block_size)[1]
        blockx = np.array(patch_block_size)[0]
        blockz = np.array(patch_block_size)[2]

        stridey = stride[1]
        stridex = stride[0]
        stridez = stride[2]

        if imagez > blockz:
            number_z = (imagez - blockz) // stridez + 1
        else:
            number_z = 1


        number_y = (y - blocky) // stridey + 1
        number_x = (x - blockx) // stridex + 1

        restored_image = np.zeros(original_shape)

                # 还原切块
        index = 0
        for i in range(number_z):
            for j in range(number_y):
                    for k in range(number_x):
                            # 计算窗口的起始和结束索引
                            start_d = i * stridez
                            end_d = start_d + blockz
                            start_h = j * stridey
                            end_h = start_h + blocky
                            start_w = k * stridex
                            end_w = start_w + blockx

                            # 将切块还原到对应位置
                            restored_image[start_d:end_d, start_h:end_h, start_w:end_w] = hr_samples[index]
                            index += 1
        
        return restored_image


    def cut_patch(self,image, sub_size, stride):
        '''
        根据切块大小和步长，补齐形状，并切成小块
            返回值数组，形状[n, patch_block_size]
        '''

        srcimg = image
        padding_z = sub_size[2]-(srcimg.shape[0]%stride[2]) if (srcimg.shape[0] % stride[2]) != 0 else 0
        padding_x = sub_size[0]-(srcimg.shape[2]%stride[0]) if (srcimg.shape[2] % stride[0]) != 0 else 0
        padding_y = sub_size[1]-(srcimg.shape[1]%stride[1]) if (srcimg.shape[1] % stride[1]) != 0 else 0
        
        srcimg_pad = np.pad(srcimg, ((0,padding_z),(0, padding_y),(0, padding_x)), mode='constant', constant_values=0)
        # print(np.shape(srcimg_pad))
        sub_images = self.subimage_generator(srcimg_pad, sub_size, stride)

        return sub_images
    
    def ImageResample_size(self, sitk_image, new_size=[128, 128, 128], is_label=False):
    
        size = np.array(sitk_image.GetSize())
        spacing = np.array(sitk_image.GetSpacing())
        new_size = np.array(new_size)
        # new_spacing = size * spacing / new_size
        new_spacing_refine = size * spacing / new_size
        new_spacing_refine = [float(s) for s in new_spacing_refine]
        new_size = [int(s) for s in new_size]

        resample = sitk.ResampleImageFilter()
        resample.SetOutputDirection(sitk_image.GetDirection())
        resample.SetOutputOrigin(sitk_image.GetOrigin())
        resample.SetSize(new_size)
        resample.SetOutputSpacing(new_spacing_refine)

        if is_label:
            resample.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            # resample.SetInterpolator(sitk.sitkBSpline)
            resample.SetInterpolator(sitk.sitkLinear)

        newimage = resample.Execute(sitk_image)
        return newimage
    
    def ImageResample_spacing(self, sitk_image, new_spacing=[1, 1, 1], is_label=False):
       
        size = np.array(sitk_image.GetSize())
        spacing = np.array(sitk_image.GetSpacing())
        new_spacing = np.array(new_spacing)
        new_spacing = [float(s) for s in new_spacing]
        new_size_refine = size * spacing / new_spacing
        new_size_refine = [int(s) for s in new_size_refine]

        resample = sitk.ResampleImageFilter()
        resample.SetOutputDirection(sitk_image.GetDirection())
        resample.SetOutputOrigin(sitk_image.GetOrigin())
        resample.SetSize(new_size_refine)
        resample.SetOutputSpacing(new_spacing)

        if is_label:
            resample.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            # resample.SetInterpolator(sitk.sitkBSpline)
            resample.SetInterpolator(sitk.sitkLinear)

        newimage = resample.Execute(sitk_image)
        return newimage
   
    def calculate_angles(self, x, y, z):
        # 归一化向量
        norm = np.sqrt(x**2 + y**2 + z**2)
        x, y, z = x / norm, y / norm, z / norm
        
        # 计算绕y轴的旋转角度
        theta_y = np.arctan2(x, z)
        theta_y_positive = np.where(theta_y < 0, theta_y + 2 * np.pi, theta_y)
        
        # 计算绕x轴的旋转角度
        theta_x = np.arctan2(y, np.sqrt(x**2 + z**2))
        theta_x_positive = np.where(theta_x < 0, theta_x + 2 * np.pi, theta_x)
        
        # 将弧度转换为角度
        theta_x_deg = np.degrees(theta_x_positive)
        theta_y_deg = np.degrees(theta_y_positive)
        
        return theta_x_deg, theta_y_deg

    def process(self, inputVolume2):
        """
        :param inputVolume2: input CT Image
        :param outputVolume: dose result
    
        """
        import numpy as np
        if not inputVolume2:
            raise ValueError("Input volume is invalid")

        import time
        startTime = time.time()
        logging.info('Processing started')
        
        # p = inputSeg.CreateBinaryLabelmapRepresentation()

        # 获取与输入 Volume 关联的 Subject Hierarchy 节点
        shNode = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
        volume1ItemID = shNode.GetItemByDataNode(inputVolume2)

        # 查找输入 Volume 的 Study 级别
        studyItemID = shNode.GetItemAncestorAtLevel(volume1ItemID, 'Study')
        # print(studyItemID)
        # ctimage几何信息
        ctimage = inputVolume2.GetImageData()
        dimension = ctimage.GetDimensions()
        spacing = inputVolume2.GetSpacing()
        origin = inputVolume2.GetOrigin()
        origin2 = [-origin[0], -origin[1], origin[2]]
        fMat = vtk.vtkMatrix4x4()
        inputVolume2.GetIJKToRASDirectionMatrix(fMat)

        min = -1000
        max = 3000
            
        ww_filter = sitk.IntensityWindowingImageFilter()
        ww_filter.SetWindowMinimum(min)
        ww_filter.SetWindowMaximum(max)
        ww_filter.SetOutputMinimum(min)
        ww_filter.SetOutputMaximum(max)
        image_ = sitkUtils.PullVolumeFromSlicer(inputVolume2)
        image_ = ww_filter.Execute(image_)

        my_image = self.ImageResample_size(image_, [128, 128, 128])

        # my_image = self.ImageResample_spacing(image_, [4, 4, 2.5])
        mydimension = my_image.GetSize()
        myspacing = my_image.GetSpacing()
        myorigin = my_image.GetOrigin()
        myorigin2 = myorigin
        # myorigin2 = [-myorigin[0], -myorigin[1], myorigin[2]]
        print('dimension', mydimension)
        print('myorigin', myorigin2)
        image_ = sitk.GetArrayFromImage(my_image)
        resampled_image = (image_ + 1000) * (255.0 / (4000.0))

        # 剂量矩阵
        blank_matrix = np.zeros(mydimension)
        blank_matrix = np.transpose(blank_matrix, (2, 1, 0))
        sub_patch_size = stride = [128, 128, 128]

        padding = [sub_patch_size[2]-(resampled_image.shape[0]%stride[2]) if (resampled_image.shape[0] % stride[2]) != 0 else 0,sub_patch_size[0]-(resampled_image.shape[2]%stride[0]) if (resampled_image.shape[2] % stride[0]) != 0 else 0, 
            sub_patch_size[1]-(resampled_image.shape[1]%stride[1]) if (resampled_image.shape[1] % stride[1]) != 0 else 0]
        new_shape = [padding[0] + resampled_image.shape[0],padding[1] + resampled_image.shape[1], padding[2] + resampled_image.shape[2]]
        # print(new_shape)

        # 筛选出 MarkupsFiducial 类型的子节点
        childPointsIDs = vtk.vtkIdList()
        shNode.GetItemChildren(volume1ItemID, childPointsIDs)
        markupsFiducialChildren = []
        for i in range(childPointsIDs.GetNumberOfIds()):
            childItemID = childPointsIDs.GetId(i)
            childNode = shNode.GetItemDataNode(childItemID)
            # 通过 GetClassName 来判断类型
            if childNode.GetClassName() == 'vtkMRMLMarkupsFiducialNode':
                markupsFiducialChildren.append(childNode)

        for markupsNode in markupsFiducialChildren:

            pointItemID = shNode.GetItemByDataNode(markupsNode)
            # 获取子节点的数量
            # numPoints = shNode.GetNumberOfItemChildren(pointItemID)
            childItemIDs = vtk.vtkIdList()
            shNode.GetItemChildren(pointItemID, childItemIDs)
            pos_n = [0, 0, 0]

            for m in range(childItemIDs.GetNumberOfIds()):
                # inputVolume.GetNthControlPointPosition(m, pos_n)
                # print('pos',pos_n)
                childItemID = childItemIDs.GetId(m)
                description_center = shNode.GetItemAttribute(childItemID, "Center")
                center = np.fromstring(description_center.strip("[]"), dtype=np.float64, sep=" ")
                description_direction = shNode.GetItemAttribute(childItemID, "Direction")
                direction = np.fromstring(description_direction.strip("[]"), dtype=np.float64, sep=" ")
                # angles = self.calculate_angles(direction[0], direction[1], direction[2])
                pos_n = [-center[0], -center[1], center[2], direction]
                # print(pos_n)

                '''点源map'''
                r = self.point_source_map(pos_n, origin, dimension, spacing)

                # i, j, k = np.meshgrid(range(dimension[2]), range(dimension[1]), range(dimension[0]), indexing='ij')

                # r = np.sqrt((origin[2] * fMat.GetElement(2, 2) + i * spacing[2] - pos_n[2] * fMat.GetElement(2, 2)) ** 2 +
                            # (origin[1] * fMat.GetElement(1, 1) + j * spacing[1] - pos_n[1] * fMat.GetElement(1, 1)) ** 2 +
                            # (origin[0] * fMat.GetElement(0, 0) + k * spacing[0] - pos_n[0] * fMat.GetElement(0, 0)) ** 2)
                # print('rmin',np.min(r))
                # print('rmax',np.max(r))
                # one seed calculation
                # g = -0.003478 * r * 0.1 + 1.03478
                # Sk = 0.1091 * 10
                cons = 1.036
                Sk = 0.98 * 0.01
                F = 1
                G0 = 100
                G = np.where(r != 0, r ** -2, 0)  
                
                r_values = np.array([1, 1.5, 2.5, 5, 7.5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100])  # 距离 (mm)
                g_values = np.array([1.020, 1.022, 1.024, 1.030, 1.020, 1, 0.935, 0.861, 0.697, 0.553, 0.425, 0.322, 0.241, 0.179, 0.134, 0.098])
                coefficients = np.polyfit(r_values, g_values, 3) # 使用3次多项式拟合
                g_r = sum(c * r**i for i, c in enumerate(reversed(coefficients)))

                dose_rate = Sk * cons * g_r * F * G * G0 * 0.01 # 0.01: cGy--Gy

                # e = math.e
                half_life = 59.4 *24 # 125I 59.4 days
                
                dose_ab = dose_rate * half_life / np.log(2)
                
                # t = 120*24 # 30days

                # d0 = np.where(r>4,15.4430*e**(-2.65 * 0.1* r), 15.4430*e**(-2.65 * 0.1* 4))

                # d0 = np.where(r>5, 0.7264/(0.1*r-0.3301)-0.1553, 0.7264/(0.1*5-0.3301)-0.1553)
                # d0 = np.where(d0<0, 0, d0)
                # dose_ab = Sk*d0*(1-e**(-t*0.623/half_life))
                print('dose_Ab',dose_ab.shape)
                print('dosemax',np.max(dose_ab))

                blank_matrix = blank_matrix + dose_ab
           
        
        # ##直接生成doseVolume的节点
        dose_ = sitk.GetImageFromArray(blank_matrix)
        dose_.SetSpacing(myspacing)
        dose_ = self.ImageResample_size(dose_, dimension)
        # dose_ = self.ImageResample_spacing(dose_, spacing)
        dose_node = slicer.util.addVolumeFromArray(sitk.GetArrayFromImage(dose_), nodeClassName="vtkMRMLScalarVolumeNode") 
        doseimage = dose_node.GetImageData()
        dosedimension = doseimage.GetDimensions()

        # print('dosedimension',dosedimension)
        # 设置属性
        dose_node.SetSpacing(spacing)
        dose_node.SetOrigin(origin)
        dose_node.SetIJKToRASDirectionMatrix(fMat)

        doseID = shNode.GetItemByDataNode(dose_node)
        shNode.SetItemParent(doseID, volume1ItemID)
        shNode.SetItemName(doseID, 'RTDoseMap')
        # # 将doseimage加入study下
        if studyItemID:
            shNode.SetItemParent(doseID, studyItemID)

         #  dose name value 在 slicer 中的属性名称, 确认所属的study 是否已经包含了这个变量
        doseUnitNameInStudy = shNode.GetItemAttribute(studyItemID,"DicomRtImport.DoseUnitName")
        doseUnitValueInStudy = shNode.GetItemAttribute(studyItemID,"DicomRtImport.DoseUnitValue")

        defaultDoseUnitName = doseUnitNameInStudy if doseUnitNameInStudy else "Gy"
        defaultDoseUnitValue = doseUnitValueInStudy if doseUnitValueInStudy else "1.0"
        # 如果study中没有dose name value

        shNode.SetItemAttribute(studyItemID, "DicomRtImport.DoseUnitName", defaultDoseUnitName)
        shNode.SetItemAttribute(studyItemID, "DicomRtImport.DoseUnitValue", defaultDoseUnitValue)

        dose_node.SetAttribute("DicomRtImport.DoseVolume", "1")

        # 请求所有者插件搜索
        shNode.RequestOwnerPluginSearch(doseID)

        # 通知项已被修改
        shNode.ItemModified(doseID)


        stopTime = time.time()
        logging.info(f'Processing completed in {stopTime-startTime:.2f} seconds')
        print(f'Processing completed in {stopTime-startTime:.2f} seconds')
