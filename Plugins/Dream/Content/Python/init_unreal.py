#Thanks to Mystfit https://github.com/Mystfit/Unreal-StableDiffusionTools/tree/master/StableDiffusionTools/Content/Python
import importlib.util
import unreal
import signal
import install_dependencies

#adding menus
menus = unreal.ToolMenus.get()

# Find the 'edit' menu, this should not fail,
# but if we're looking for a menu we're unsure about 'if not'
# works as nullptr check,

main_menu = menus.find_menu("LevelEditor.MainMenu")
my_menu = main_menu.add_sub_menu("[My.Menu](https://github.com/albertotrunk/UE5-Dream)", "Python", "My Menu", "Let's Dream")

for name in ["Dream", "Install"]:
    e = unreal.ToolMenuEntry(
        name = name,
        type = unreal.MultiBlockType.MENU_ENTRY, # If you pass a type that is not supported Unreal will let you know,
    )
    e.set_label(name)
    if name == "Dream":
        e.set_string_command(
            type=unreal.ToolMenuStringCommandType.PYTHON,
            custom_type=name,
            string="import unreal;unreal.EditorUtilitySubsystem().spawn_and_register_tab(unreal.EditorAssetLibrary.load_asset('/Dream/dreamUI.dreamUI'))"       #< !! This is where things get interesting
        )
    if name == "Install":
        e.set_string_command(
            type=unreal.ToolMenuStringCommandType.PYTHON,
            custom_type=name,
            string="install_dependencies.py"       #< !! This is where things get interesting
        )

    my_menu.add_menu_entry("Items", e)



menus.refresh_all_widgets()
#------------------------------------------------------------------------------


# Replace print() command to fix Unreal flagging every Python print call as an error
print = unreal.log

# Redirect missing SIGKILL signal on windows to SIGTERM
signal.SIGKILL = signal.SIGTERM


def SD_dependencies_installed():
    dependencies = install_dependencies.dependencies
    installed = True
    modules = [package_opts["module"] if "module" in package_opts else package_name for package_name, package_opts in dependencies.items()]
    for module in modules:
        print(f"Looking for module {module}")
        try:
            if not importlib.util.find_spec(module):
                raise(ValueError())
        except ValueError:
            print("Missing Stable Diffusion dependency {0}. Please install or update the plugin's python dependencies".format(module))
            installed = False

    return installed


# Check if dependencies are installed correctly
dependencies_installed = SD_dependencies_installed()
print("Stable Diffusion dependencies are {0}available".format("" if dependencies_installed else "not "))

if SD_dependencies_installed:
    try:
        import load_diffusers_bridge
    except ImportError:
        print("Skipping default Diffusers Bridge load until dependencies have been installed")
