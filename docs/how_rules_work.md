# Cómo se Aplican las Reglas de Interpretación

## Flujo General

1. **Carga de reglas**: `interpretation_config.py` carga el JSON desde `rules/interpretation_rules.json`
2. **Aplicación**: Los interpretadores usan las funciones helper para obtener las reglas
3. **Evaluación**: Se evalúan las condiciones en orden y se toma la primera que coincida

## Ejemplo: Swing

### Regla en JSON:
```json
{
  "swing": {
    "field": "swing_ratio",
    "has_swing_field": "has_swing",
    "priority_condition": "has_swing == false",
    "priority_level": "straight",
    "priority_description": "No swing detected...",
    "thresholds": [
      {"max": 1.3, "level": "light", "description": "..."},
      {"max": 1.5, "level": "moderate", "description": "..."},
      {"max": null, "level": "heavy", "description": "..."}
    ]
  }
}
```

### Cómo se aplica:

```python
from config.interpretation_config import get_swing_rules, find_threshold_level

def interpret_swing(swing_data: dict[str, Any]) -> SwingInterpretation:
    rules = get_swing_rules()
    ratio = swing_data.get(rules["field"], 1.0)
    has_swing = swing_data.get(rules["has_swing_field"], False)
    
    # 1. Verificar condición de prioridad
    if not has_swing:  # priority_condition: has_swing == false
        level = SwingLevel(rules["priority_level"])
        description = rules["priority_description"]
    else:
        # 2. Buscar threshold que coincida con el ratio
        threshold = find_threshold_level(ratio, rules["thresholds"])
        if threshold is None:
            threshold = rules["thresholds"][-1]  # Último (default)
        level = SwingLevel(threshold["level"])
        description = threshold["description"]
    
    return SwingInterpretation(level=level, ratio=ratio, description=description)
```

## Ejemplo: Brightness

### Regla en JSON:
```json
{
  "brightness": {
    "field": "brightness_value",
    "thresholds": [
      {"max": 0.15, "level": "dark", "description": "..."},
      {"max": 0.25, "level": "warm", "description": "..."},
      ...
    ]
  }
}
```

### Cómo se aplica:

```python
from config.interpretation_config import get_brightness_rules, find_threshold_level

def interpret_brightness(brightness_value: float) -> BrightnessInterpretation:
    rules = get_brightness_rules()
    
    # Buscar threshold que coincida
    threshold = find_threshold_level(brightness_value, rules["thresholds"])
    if threshold is None:
        threshold = rules["thresholds"][-1]  # Default
    
    level = BrightnessLevel(threshold["level"])
    description = threshold["description"]
    
    return BrightnessInterpretation(level=level, ...)
```

## Ejemplo: Harmonicity (con orden de evaluación)

### Regla en JSON:
```json
{
  "harmonicity": {
    "evaluation_order": [
      {"name": "pure_harmonic", "condition": "harmonic_ratio >= 0.85", ...},
      {"name": "pure_percussive", "condition": "percussive_ratio >= 0.85", ...},
      ...
    ]
  }
}
```

### Cómo se aplica:

```python
def interpret_harmonicity(harmonic_data: dict) -> HarmonicityInterpretation:
    rules = get_harmonicity_rules()
    harmonic_ratio = harmonic_data.get("harmonic_ratio", 0.5)
    percussive_ratio = harmonic_data.get("percussive_ratio", 0.5)
    
    # Evaluar en orden
    for rule in rules["evaluation_order"]:
        if rule["condition"] == "default":
            # Última regla (default)
            level = HarmonicityLevel(rule["level"])
            description = rule["description"]
            break
        elif eval_condition(rule["condition"], harmonic_ratio, percussive_ratio):
            level = HarmonicityLevel(rule["level"])
            description = rule["description"]
            break
    
    return HarmonicityInterpretation(...)
```

## Estado Actual

**Problema**: Los interpretadores (`rhythm_interpreter.py` y `tone_color_interpreter.py`) tienen código hardcodeado y NO están usando las reglas del JSON.

**Solución**: Actualizar los interpretadores para que:
1. Importen las funciones de `config.interpretation_config`
2. Usen `get_*_rules()` para obtener las reglas
3. Usen `find_threshold_level()` para encontrar el threshold correcto
4. Evalúen condiciones en el orden especificado

