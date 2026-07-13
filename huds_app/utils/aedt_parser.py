import re


def parse_aedt_designs(aedt_path):
    """Parse .aedt file to extract design names from DefInfo section.
    
    Args:
        aedt_path: Path to the .aedt file or directory containing it
        
    Returns:
        List of design name strings
    """
    import os
    
    if os.path.isdir(aedt_path):
        aedt_files = [f for f in os.listdir(aedt_path) if f.endswith('.aedt')]
        if not aedt_files:
            return []
        aedt_file = os.path.join(aedt_path, aedt_files[0])
    else:
        aedt_file = aedt_path
    
    if not os.path.exists(aedt_file):
        return []
    
    try:
        with open(aedt_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception:
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
    
    return list(dict.fromkeys(designs))  # Deduplicate preserving order


def parse_aedt_variables(aedt_path, design_name):
    """Parse .aedt file to extract variable definitions for a specific design.
    
    Args:
        aedt_path: Path to the .aedt file or directory containing it
        design_name: Name of the design to extract variables from
        
    Returns:
        List of dicts with keys: name, default, unit
    """
    import os
    
    if os.path.isdir(aedt_path):
        aedt_files = [f for f in os.listdir(aedt_path) if f.endswith('.aedt')]
        if not aedt_files:
            return []
        aedt_file = os.path.join(aedt_path, aedt_files[0])
    else:
        aedt_file = aedt_path
    
    if not os.path.exists(aedt_file):
        return []
    
    try:
        with open(aedt_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception:
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
    
    return unique_vars


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
