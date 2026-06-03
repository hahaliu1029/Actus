"""Permission engine subpackage.

All approval / deny / ask decisions for tool calls flow through
PermissionEngine; ApprovalStateWriter (R5 CS4) is called
only from within DefaultPermissionEngine.

INV-1b (CI gate): writer.write / write_audit_only / delete_grant outside
this package or its DI factory fails the build.
"""
