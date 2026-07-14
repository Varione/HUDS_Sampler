import re


def _resolve_aedt_file(aedt_path):
    """Resolve directory to .aedt file, matching by name if multiple exist."""
    import os
    if not os.path.isdir(aedt_path):
        return aedt_path
    aedt_files = [f for f in os.listdir(aedt_path) if f.endswith('.aedt')]
    if not aedt_files:
        return None
    if len(aedt_files) == 1:
        return os.path.join(aedt_path, aedt_files[0])
    # Match by directory name
    dir_name = os.path.basename(os.path.normpath(aedt_path))
    for f in aedt_files:
        if f.startswith(dir_name):
            return os.path.join(aedt_path, f)
    return os.path.join(aedt_path, aedt_files[0])


def parse_aedt_designs(aedt_path):
    """Parse .aedt file to extract design names from DefInfo section.
    
    Args:
        aedt_path: Path to the .aedt file or directory containing it
        
    Returns:
        List of design name strings
    """
    import os

    aedt_file = _resolve_aedt_file(aedt_path)
    if not aedt_file:
        return []

    try:
        with open(aedt_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        print(f"[AEDT] parse_designs: Read {len(lines)} lines from {aedt_file}")
    except Exception as e:
        print(f"[AEDT] Failed to read file: {e}")
        return []
    
    designs = []
    in_definfo = False
    
    for line in lines:
        stripped = line.strip()
        
        # Track DefInfo section which lists all designs
        if "$begin 'DefInfo'" in stripped:
            in_definfo = True
            continue
            
        if "$end 'DefInfo'" in stripped:
            in_definfo = False
            continue
        
        # Inside DefInfo, extract design names like: ly(1002, 0, ...)
        if in_definfo:
            match = re.match(r"(\w+)\s*\(", stripped)
            if match:
                designs.append(match.group(1))
    
    print(f"[AEDT] parse_designs: Found {len(designs)} designs")
    return list(dict.fromkeys(designs))  # Deduplicate preserving order


def _parse_all_model_designs(lines):
    """Extract design names from Maxwell3DModel blocks (first Name= inside each block)."""
    designs = []
    in_model = False
    found_name = False
    for line in lines:
        stripped = line.strip()
        if "$begin 'Maxwell3DModel'" in stripped:
            in_model = True
            found_name = False
            continue
        if "$end 'Maxwell3DModel'" in stripped:
            in_model = False
            continue
        if in_model and not found_name:
            match = re.match(r"Name='([^']+)'", stripped)
            if match:
                designs.append(match.group(1))
                found_name = True
    return designs


def parse_aedt_variables(aedt_path, design_name):
    """Parse .aedt file to extract variable definitions for a specific design.
    
    Args:
        aedt_path: Path to the .aedt file or directory containing it
        design_name: Name of the design to extract variables from
        
    Returns:
        List of dicts with keys: name, default, unit
    """
    import os

    aedt_file = _resolve_aedt_file(aedt_path)
    if not aedt_file:
        return []

    try:
        with open(aedt_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        print(f"[AEDT] parse_variables for '{design_name}': Read {len(lines)} lines from {aedt_file}")
    except Exception as e:
        print(f"[AEDT] Failed to read file: {e}")
        return []

    current_design = None
    in_properties = False
    variables = []
    
    for line in lines:
        stripped = line.strip()
        
        # Track Maxwell3DModel blocks to identify design sections
        if "$begin 'Maxwell3DModel'" in stripped:
            current_design = None  # Reset when entering new model block
            continue
            
        if "$end 'Maxwell3DModel'" in stripped:
            current_design = None
            continue
        
        # Inside Maxwell3DModel, detect design name
        if current_design is None and stripped.startswith("Name='"):
            match = re.match(r"Name='([^']+)'", stripped)
            if match and match.group(1) == design_name:
                current_design = design_name
        
        # Detect Properties section within target design
        if current_design and "$begin 'Properties'" in stripped:
            in_properties = True
        elif current_design and "$end 'Properties'" in stripped:
            in_properties = False
        
        # Parse VariableProp within Properties section for target design
        if in_properties and current_design == design_name and "VariableProp(" in line:
            match = re.search(r"VariableProp\('([^']+)',\s*'([^']+)',\s*'([^']*)',\s*'([^']+)'", line)
            if match:
                name, vtype, desc, default = match.groups()
                # Extract unit from default value
                _, unit = parse_value_with_unit(default)
                variables.append({
                    'name': name,
                    'default': default,
                    'unit': unit,
                })
    
    # Deduplicate by name
    seen = set()
    unique_vars = []
    for v in variables:
        if v['name'] not in seen:
            seen.add(v['name'])
            unique_vars.append(v)
    
    # Fallback: if exact match yielded nothing, use first available design
    if not unique_vars:
        all_designs = _parse_all_model_designs(lines)
        if all_designs:
            fallback_name = all_designs[0]
            print(f"[AEDT] parse_variables: exact match for '{design_name}' yielded 0 results, falling back to '{fallback_name}'")
            return parse_aedt_variables(aedt_path, fallback_name)
    
    print(f"[AEDT] parse_variables: Found {len(unique_vars)} variables for '{design_name}'")
    return unique_vars


def parse_aedt_outputs(aedt_path, design_name):
    """Parse .aedt file to extract output variable names (Maxwell parameters).
    
    Args:
        aedt_path: Path to the .aedt file or directory containing it
        design_name: Name of the design
        
    Returns:
        List of dicts with keys: name, type (e.g., 'Force', 'Torque')
    """
    import os

    aedt_file = _resolve_aedt_file(aedt_path)
    if not aedt_file:
        return []

    try:
        with open(aedt_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        print(f"[AEDT] parse_outputs for '{design_name}': Read {len(lines)} lines from {aedt_file}")
    except Exception as e:
        print(f"[AEDT] Failed to read file: {e}")
        return []
    
    current_design = None
    in_maxwell_params = False
    in_sim_data_extractor = False
    current_param_name = None
    outputs = []
    
    for line in lines:
        stripped = line.strip()
        
        # Track Maxwell3DModel blocks
        if "$begin 'Maxwell3DModel'" in stripped:
            current_design = None
            continue
            
        if "$end 'Maxwell3DModel'" in stripped:
            current_design = None
            continue
        
        # Detect design name
        if current_design is None and stripped.startswith("Name='"):
            match = re.match(r"Name='([^']+)'", stripped)
            if match and match.group(1) == design_name:
                current_design = design_name
        
        if not current_design:
            continue
        
        # Track MaxwellParameterSetup section
        if "$begin 'MaxwellParameters'" in stripped:
            in_maxwell_params = True
            continue
            
        if "$end 'MaxwellParameters'" in stripped and in_maxwell_params:
            in_maxwell_params = False
            continue
        
        # Track SimDataExtractor section
        if "$begin 'SimDataExtractor'" in stripped:
            in_sim_data_extractor = True
            continue
            
        if "$end 'SimDataExtractor'" in stripped:
            in_sim_data_extractor = False
            continue
        
        # Parse Maxwell parameter definitions
        if in_maxwell_params:
            param_match = re.match(r"\$begin '(\w+)'", stripped)
            if param_match:
                current_param_name = param_match.group(1)
                continue
            
            type_match = re.search(r"MaxwellParameterType='([^']+)'", stripped)
            if type_match and current_param_name:
                outputs.append({
                    'name': current_param_name,
                    'type': type_match.group(1),
                })
        
        # Parse SimValue entries for actual output quantities
        if in_sim_data_extractor:
            sim_match = re.search(r"SimValue\('([^']+)'", stripped)
            if sim_match:
                full_name = sim_match.group(1)
                outputs.append({
                    'name': full_name,
                    'type': 'Result',
                })
    
    # Deduplicate by name
    seen = set()
    unique_outputs = []
    for o in outputs:
        if o['name'] not in seen:
            seen.add(o['name'])
            unique_outputs.append(o)
    
    # Fallback: if exact match yielded nothing, use first available design
    if not unique_outputs:
        all_designs = _parse_all_model_designs(lines)
        if all_designs:
            fallback_name = all_designs[0]
            print(f"[AEDT] parse_outputs: exact match for '{design_name}' yielded 0 results, falling back to '{fallback_name}'")
            return parse_aedt_outputs(aedt_path, fallback_name)
    
    print(f"[AEDT] parse_outputs: Found {len(unique_outputs)} outputs for '{design_name}'")
    return unique_outputs


def parse_value_with_unit(value_str):
    """Parse a value string like '180mm' into numeric value and unit.
    
    Returns:
        Tuple of (numeric_value, unit_string) or (None, original_string) if parsing fails
    """
    if not value_str:
        return None, ''
    
    # Handle expressions like '0.5/(v/3.6)'
    if '/' in value_str or '(' in value_str:
        return None, value_str
    
    match = re.match(r'^([+-]?\d*\.?\d+)\s*(.+)$', value_str)
    if match:
        try:
            num = float(match.group(1))
            unit = match.group(2)
            return num, unit
        except ValueError:
            pass
    
    # Try pure numeric
    try:
        return float(value_str), ''
    except ValueError:
        pass
    
    return None, value_str
