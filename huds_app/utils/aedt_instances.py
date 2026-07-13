import subprocess
import win32com.client


def _get_running_pids():
    """Get PIDs of all running ansysedt.exe processes."""
    pids = []
    try:
        result = subprocess.run(
            ['tasklist', '/FI', 'IMAGENAME eq ansysedt.exe', '/FO', 'CSV', '/NH'],
            capture_output=True, text=True
        )
        if not result.stdout.strip():
            return pids
        for line in result.stdout.strip().split('\n'):
            parts = line.replace('"', '').split(',')
            if len(parts) >= 2:
                try:
                    pids.append(int(parts[1]))
                except ValueError:
                    continue
    except Exception:
        pass
    return pids


def _try_connect(prog_id):
    """Try to connect to AEDT via a specific ProgID.
    
    Returns (oApp, oDesktop, version, projects) or None on failure.
    """
    try:
        oApp = win32com.client.Dispatch(prog_id)
        oDesktop = oApp.GetAppDesktop()
        version = oDesktop.GetVersion()
        projects = []
        try:
            projs = oDesktop.GetProjects()
            for i in range(projs.Count):
                projects.append(projs(i).GetName())
        except Exception:
            pass
        return (oApp, oDesktop, version, projects)
    except Exception:
        return None


def _instances_are_same(inst1, inst2):
    """Check if two instance dicts represent the same AEDT instance."""
    if not inst1 or not inst2:
        return False
    return (inst1['version'] == inst2['version']
            and inst1['projects'] == inst2['projects'])


def enumerate_aedt_instances():
    """Enumerate all running AEDT COM instances.
    
    Uses multiple strategies to find all running instances:
    1. Try versioned ProgIDs (Dispatch creates/connects to instances)
    2. Try GetActiveObject for each versioned ProgID
    3. Try GetRunningApps/GetRunningDesktops API
    
    Returns:
        List of dicts with keys:
            - oApp: COM application object
            - oDesktop: COM desktop object
            - version: version string (e.g., '2025.1')
            - projects: list of open project names
            - label: human-readable label for display
    """
    prog_ids = [
        "Ansoft.ElectronicsDesktop.2025.1",
        "Ansoft.ElectronicsDesktop.2024.1",
        "Ansoft.ElectronicsDesktop.2023.1",
        "Ansoft.ElectronicsDesktop.2022.1",
        "Ansoft.ElectronicsDesktop.2021.1",
    ]

    connected = []

    # Strategy 1: Dispatch through versioned ProgIDs
    for prog_id in prog_ids:
        result = _try_connect(prog_id)
        if result:
            oApp, oDesktop, version, projects = result
            inst = {
                'oApp': oApp,
                'oDesktop': oDesktop,
                'version': version,
                'projects': projects,
                'label': f"AEDT {version}  ({len(projects)} projects)",
            }
            if not any(_instances_are_same(c, inst) for c in connected):
                connected.append(inst)

    # Strategy 2: GetActiveObject for each ProgID
    for prog_id in prog_ids:
        try:
            oApp = win32com.client.GetActiveObject(prog_id)
            oDesktop = oApp.GetAppDesktop()
            version = oDesktop.GetVersion()
            projects = []
            try:
                projs = oDesktop.GetProjects()
                for i in range(projs.Count):
                    projects.append(projs(i).GetName())
            except Exception:
                pass
            inst = {
                'oApp': oApp,
                'oDesktop': oDesktop,
                'version': version,
                'projects': projects,
                'label': f"AEDT {version}  ({len(projects)} projects) [Active]",
            }
            if not any(_instances_are_same(c, inst) for c in connected):
                connected.append(inst)
        except Exception:
            continue

    # Strategy 3: GetRunningApps / GetRunningDesktops
    if connected:
        first = connected[0]
        try:
            oDesktop = first['oDesktop']
            get_running = getattr(oDesktop, 'GetRunningApps', None) or \
                          getattr(first['oApp'], 'GetRunningApps', None)
            if get_running:
                apps = get_running()
                for i in range(len(apps)):
                    app = apps(i)
                    try:
                        desktop = app.GetAppDesktop()
                        version = desktop.GetVersion()
                        projects = []
                        projs = desktop.GetProjects()
                        for j in range(projs.Count):
                            projects.append(projs(j).GetName())
                        inst = {
                            'oApp': app,
                            'oDesktop': desktop,
                            'version': version,
                            'projects': projects,
                            'label': f"AEDT {version}  ({len(projects)} projects) [Running]",
                        }
                        if not any(_instances_are_same(c, inst) for c in connected):
                            connected.append(inst)
                    except Exception:
                        continue
        except Exception:
            pass

    # If still only one but multiple PIDs, add numbered labels
    pids = _get_running_pids()
    if len(pids) > 1 and len(connected) == 1:
        connected[0]['label'] = f"AEDT {connected[0]['version']}  ({len(pids)} processes running)"

    return connected


def connect_to_aedt_instance(index=0):
    """Connect to a specific AEDT instance by index.
    
    Args:
        index: Index of the instance to connect to
        
    Returns:
        Tuple of (oApp, oDesktop, version, projects) or (None, None, '', [])
    """
    instances = enumerate_aedt_instances()
    if 0 <= index < len(instances):
        inst = instances[index]
        return (inst['oApp'], inst['oDesktop'],
                inst['version'], inst['projects'])
    return None, None, '', []


def get_first_aedt():
    """Connect to the first available AEDT instance (legacy behavior).
    
    Returns:
        Tuple of (oApp, oDesktop) or (None, None)
    """
    instances = enumerate_aedt_instances()
    if instances:
        return instances[0]['oApp'], instances[0]['oDesktop']
    return None, None
