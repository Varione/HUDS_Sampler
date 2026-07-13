import re


def parse_aedt_variables(aedt_path, design_name):
    """Parse .aedt file to extract variable definitions for a specific design.
    
    Args:
        aedt_path: Path to the .aedt file (directory or file)
        design_name: Name of the design to extract variables from
        
    Returns:
        List of dicts with keys: name, default, min, max (min/max may be None)
    """
    import os
    
    # Handle directory path
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
        # Detect design section (Maxwell3DModel with Name=)
        if "Name='" in line and any(c in line for c in ['ly', 'Design']):
            match = re.search(r"Name='([^']*)'", line)
            if match and match.group(1) == design_name:
                current_design = design_name
        
        # Detect Properties section
        if "$begin 'Properties'" in line:
            in_properties = True
        elif "$end 'Properties'" in line:
            in_properties = False
        
        # Parse VariableProp within Properties section for target design
        if in_properties and current_design == design_name and "VariableProp(" in line:
            match = re.search(r"VariableProp\('([^']+)',\s*'([^']+)',\s*'([^']*)',\s*'([^']*)'(?:,\s*oa\(([^)]*)\))?", line)
            if match:
                name, vtype, desc, default, opt_info = match.groups()
                var_info = {'name': name, 'default': default}
                if opt_info:
                    min_match = re.search(r"Min='([^']+)'", opt_info)
                    max_match = re.search(r"Max='([^']+)'", opt_info)
                    if min_match:
                        var_info['min'] = min_match.group(1)
                    if max_match:
                        var_info['max'] = max_match.group(1)
                variables.append(var_info)
    
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
