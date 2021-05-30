We use python files as configs, incorporate modular and inheritance design into our config system, which is convenient to conduct various experiments. You can find all the provided configs under $MMPose/configs. If you wish to inspect the config file, you may run python tools/analysis/print_config.py /PATH/TO/CONFIG to see the complete config.

Modify config through script arguments
Config File Naming Convention
Config System Example
FAQ
Use intermediate variables in configs
Modify config through script arguments
When submitting jobs using “tools/train.py” or “tools/test.py”, you may specify --cfg-options to in-place modify the config.

Update config keys of dict chains.

The config options can be specified following the order of the dict keys in the original config. For example, --cfg-options model.backbone.norm_eval=False changes the all BN modules in model backbones to train mode.

Update keys inside a list of configs.

Some config dicts are composed as a list in your config. For example, the training pipeline data.train.pipeline is normally a list e.g. [dict(type='LoadImageFromFile'), dict(type='TopDownRandomFlip', flip_prob=0.5), ...]. If you want to change 'flip_prob=0.5' to 'flip_prob=0.0' in the pipeline, you may specify --cfg-options data.train.pipeline.1.flip_prob=0.0.

Update values of list/tuples.

If the value to be updated is a list or a tuple. For example, the config file normally sets workflow=[('train', 1)]. If you want to change this key, you may specify --cfg-options workflow="[(train,1),(val,1)]". Note that the quotation mark “ is necessary to support list/tuple data types, and that NO white space is allowed inside the quotation marks in the specified value.

Config File Naming Convention
We follow the style below to name config files. Contributors are advised to follow the same style.

configs/{task}/{model}/{dataset}/{backbone}_[model_setting]_{dataset}_{input_size}_[technique].py
{xxx} is required field and [yyy] is optional.

{task}: method type, e.g. top_down, bottom_up, hand, mesh, etc.
{model}: model type, e.g. hrnet, darkpose, etc.
{dataset}: dataset name, e.g. coco, etc.
{backbone}: backbone type, e.g. res50 (ResNet-50), etc.
[model setting]: specific setting for some models.
[misc]: miscellaneous setting/plugins of model, e.g. video, etc.
{input_size}: input size of the model.
[technique]: some specific techniques or tricks to use, e.g. dark, udp.
Config System for Top-down Human Pose Estimation
An Example of 2D Top-down Human Pose Estimation

To help the users have a basic idea of a complete config structure and the modules in the config system, we make brief comments on ‘configs/top_down/resnet/coco/res50_coco_256x192.py’ as the following. For more detailed usage and alternative for per parameter in each module, please refer to the API documentation.
