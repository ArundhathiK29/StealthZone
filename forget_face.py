"""
forget_face.py

A utility to remove a registered face from the StealthZone application.
Implements the "forget me" privacy feature.

Usage:
    python forget_face.py "Name to Remove"
    
    Or run without arguments to see a list and select interactively.
"""
import sys
from face_registry import load_registry, save_registry

def list_registered_faces():
    """Display all registered faces."""
    registry = load_registry()
    if not registry:
        print("\n❌ No registered faces found.")
        return None
    
    print("\n📋 Currently registered faces:")
    print("=" * 40)
    for idx, name in enumerate(registry.keys(), 1):
        print(f"  {idx}. {name}")
    print("=" * 40)
    return registry

def remove_face(name_to_remove):
    """Remove a specific face from the registry."""
    registry = load_registry()
    
    if not registry:
        print("\n❌ No registered faces found.")
        return False
    
    if name_to_remove not in registry:
        print(f"\n❌ '{name_to_remove}' is not in the registry.")
        print(f"\nAvailable names: {', '.join(registry.keys())}")
        return False
    
    # Confirm deletion
    confirmation = input(f"\n⚠️  Are you sure you want to remove '{name_to_remove}'? (yes/no): ").strip().lower()
    
    if confirmation in ['yes', 'y']:
        del registry[name_to_remove]
        save_registry(registry)
        print(f"\n✅ Successfully removed '{name_to_remove}' from the registry.")
        return True
    else:
        print("\n❌ Deletion cancelled.")
        return False

def main():
    print("\n" + "=" * 50)
    print("  StealthZone - Forget Face Utility")
    print("  'Right to be Forgotten' Privacy Tool")
    print("=" * 50)
    
    # Check if name was provided as command-line argument
    if len(sys.argv) > 1:
        name = ' '.join(sys.argv[1:])
        remove_face(name)
    else:
        # Interactive mode
        registry = list_registered_faces()
        
        if registry:
            print("\nOptions:")
            print("  1. Enter a name to remove")
            print("  2. Enter a number from the list")
            print("  3. Type 'all' to clear the entire registry")
            print("  4. Type 'quit' to exit")
            
            choice = input("\nYour choice: ").strip()
            
            if choice.lower() == 'quit':
                print("\n👋 Exiting without changes.")
                return
            
            elif choice.lower() == 'all':
                confirmation = input("\n⚠️  Are you sure you want to DELETE ALL registered faces? (yes/no): ").strip().lower()
                if confirmation in ['yes', 'y']:
                    registry.clear()
                    save_registry(registry)
                    print("\n✅ All faces have been removed from the registry.")
                else:
                    print("\n❌ Deletion cancelled.")
            
            elif choice.isdigit():
                idx = int(choice) - 1
                names = list(registry.keys())
                if 0 <= idx < len(names):
                    remove_face(names[idx])
                else:
                    print("\n❌ Invalid number.")
            
            else:
                remove_face(choice)
    
    print("\n" + "=" * 50 + "\n")

if __name__ == "__main__":
    main()
