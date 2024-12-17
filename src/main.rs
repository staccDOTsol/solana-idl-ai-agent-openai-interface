use anchor_lang::{prelude::*};
use axum::routing::{post, get};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{fmt::Display, str::FromStr, net::SocketAddr};
use axum::{BoxError, Json, Router};
use axum::extract::Path;
use axum::handler::Handler;
use utoipa::OpenApi;
use utoipa::ToSchema;
use utoipa_swagger_ui::SwaggerUi;

#[derive(Deserialize, Clone, Serialize, ToSchema)]
struct IdlField {
    name: String,
    #[serde(flatten)]
    type_info: IdlTypeVariant,
}

#[derive(Deserialize, Clone, Serialize, ToSchema, Debug)]
#[serde(untagged)]
enum IdlTypeVariant {
    Detailed { type_info: IdlTypeInfo },
    Simple { r#type: String },
    Defined { r#type: Value },
}

#[derive(Deserialize, Serialize, Clone, ToSchema, Debug)]
struct IdlTypeInfo {
    kind: String, // "defined", "primitive", etc.
    defined_type: Option<String>, // For user-defined types
    primitive_type: Option<String>, // For primitive types like u64, string, etc.
    array_length: Option<u32>, // For fixed-length arrays
    vec_type: Option<Box<IdlTypeInfo>>, // For vectors
}

#[derive(Deserialize, Clone, Serialize, ToSchema)]
struct IdlProgramSeed {
    kind: String,
    #[serde(flatten)]  // Similar to what we did with IdlSeed
    value: Value,  // This will capture any additional fields
}

impl Display for IdlProgramSeed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Convert the value to a string representation
        let value_str = serde_json::to_string(&self.value).unwrap_or_default();
        write!(f, "{}", value_str)
    }
}

#[derive(Deserialize, Clone, Serialize, ToSchema)]
struct IdlMetadata {
    name: String,
    version: String,
    spec: String,
    description: Option<String>,
}

#[derive(Deserialize, Clone, Serialize, ToSchema)]
struct IdlAccount {
    name: String,
    discriminator: [u8; 8],
}

#[derive(Deserialize, Clone, Serialize, ToSchema)]
struct IdlType {
    name: String,
    #[serde(rename = "type")]
    type_def: Value,  // Changed from type_info to handle the actual JSON structure
}

// Generic IDL structures that can parse any Anchor IDL
#[derive(Deserialize, Clone, Serialize, ToSchema)]
struct AnchorIdl {
    address: String,
    metadata: IdlMetadata,
    instructions: Vec<IdlInstruction>,
    accounts: Vec<IdlAccount>,
    types: Vec<IdlType>,
}
use std::collections::HashMap;
use std::sync::Mutex;
use once_cell::sync::Lazy;

static CACHE: Lazy<Mutex<HashMap<String, AnchorIdl>>> = Lazy::new(|| {
    Mutex::new(HashMap::new())
});

impl AnchorIdl {
    pub fn load_cached(program_id: &str) -> Result<Self> {
      
        // Check cache first
        if let Some(cached_idl) = CACHE.lock().unwrap().get(program_id) {
            return Ok(cached_idl.clone());
        }

        // If not in cache, load from file system
        let idl_path = format!("idls/{}.json", program_id);
        let idl_json = std::fs::read_to_string(idl_path)?;
        let idl: AnchorIdl = serde_json::from_str(&idl_json).unwrap();
        
        // Store in cache
        CACHE.lock().unwrap().insert(program_id.to_string(), idl.clone());
        
        Ok(idl)
    }
}
#[derive(Deserialize, Clone, Serialize, ToSchema)]
struct IdlInstruction {
    name: String,
    discriminator: [u8; 8],
    accounts: Vec<IdlAccountItem>,
    args: Vec<IdlField>,
}

#[derive(Deserialize, Clone, Serialize, ToSchema)]
struct IdlAccountItem {
    name: String,
    writable: Option<bool>,
    signer: Option<bool>,
    pda: Option<IdlPda>,
}

#[derive(Deserialize, Clone, Serialize, ToSchema)]
struct IdlPda {
    seeds: Vec<IdlSeed>,
    program: Option<IdlProgramSeed>,
}

#[derive(Deserialize, Clone, Serialize, ToSchema)]
struct IdlSeed {
    kind: String,
    #[serde(flatten)]
    value: Value,
}

#[derive(Serialize, ToSchema)]
struct ArgumentSpec {
    name: String,
    type_info: IdlTypeInfo,
}

// API Response types
#[derive(Serialize, ToSchema)]
struct InstructionTemplate {
    name: String,
    required_accounts: Vec<RequiredAccount>,
    arguments: Vec<ArgumentSpec>,
    pda_derivations: Vec<PdaDerivation>,
}

#[derive(Serialize, ToSchema)]
struct RequiredAccount {
    name: String,
    is_signer: bool,
    is_writable: bool,
}

#[derive(Serialize, ToSchema)]
struct PdaDerivation {
    account_name: String,
    seeds: Vec<SeedComponent>,
    program_id: Option<String>,
}

#[derive(Serialize, ToSchema)]
struct SeedComponent {
    kind: String,
    value: Option<Vec<u8>>,
    depends_on: Option<String>,
}

impl AnchorIdl {
    // Process any IDL into instruction templates
    pub fn process_idl(idl_json: &str) -> Vec<InstructionTemplate> {
        let idl: AnchorIdl = serde_json::from_str(idl_json).unwrap();
        // Save the IDL to the filesystem
        std::fs::create_dir_all("idls").unwrap();
        let idl_path = format!("idls/{}.json", idl.address);
        std::fs::write(&idl_path, idl_json).unwrap();
        idl.instructions.iter().map(|ix| {
            let required_accounts = ix.accounts.iter().map(|acc| {
                RequiredAccount {
                    name: acc.name.clone(),
                    is_signer: acc.signer.unwrap_or(false),
                    is_writable: acc.writable.unwrap_or(false),
                }
            }).collect();

            let pda_derivations = ix.accounts.iter()
                .filter_map(|acc| acc.pda.as_ref().map(|pda| {
                    PdaDerivation {
                        account_name: acc.name.clone(),
                        seeds: pda.seeds.iter().map(|seed| {
                            match seed.kind.as_str() {
                                "const" => SeedComponent {
                                    kind: "const".to_string(),
                                    value: seed.value.get("value")
                                        .and_then(|v| v.as_array())
                                        .map(|arr| arr.iter()
                                            .filter_map(|v| v.as_u64())
                                            .map(|n| n as u8)
                                            .collect()),
                                    depends_on: None,
                                },
                                "account" => SeedComponent {
                                    kind: "account".to_string(),
                                    value: None,
                                    depends_on: seed.value.get("path")
                                        .and_then(|v| v.as_str())
                                        .map(String::from),
                                },
                                _ => panic!("Unknown seed kind"),
                            }
                        }).collect(),
                        program_id: pda.program.clone().map(|p| p.to_string()),
                    }
                }))
                .collect();

            InstructionTemplate {
                name: ix.name.clone(),
                required_accounts,
                arguments: ix.args.iter().map(|arg| ArgumentSpec {
                    name: arg.name.clone(),
                    type_info: match &arg.type_info {
                        IdlTypeVariant::Detailed { type_info } => type_info.clone(),
                        IdlTypeVariant::Simple { r#type } => IdlTypeInfo {
                            kind: "primitive".to_string(),
                            primitive_type: Some(r#type.clone()),
                            defined_type: None,
                            array_length: None,
                            vec_type: None,
                        },
                        IdlTypeVariant::Defined { r#type } => IdlTypeInfo {
                            kind: "defined".to_string(),
                            defined_type: Some(r#type.get("defined").and_then(|v| v.get("name")).and_then(|v| v.as_str()).unwrap_or_default().to_string()),
                            primitive_type: None,
                            array_length: None,
                            vec_type: None,
                        },
                    },
                }).collect(),
                pda_derivations,
            }
        }).collect()
        
    }

    // Generate actual instruction data
    pub fn generate_instruction(
        &self,
        ix_name: &str,
        accounts: Vec<AccountMeta>,
        args: Vec<Value>
    ) -> Instruction {
        let ix = self.instructions.iter()
            .find(|i| i.name == ix_name)
            .expect("Instruction not found");

        let mut data = ix.discriminator.to_vec();
        
        // Serialize arguments directly into the data vector
        for (arg, arg_def) in args.iter().zip(ix.args.iter()) {
            match &arg_def.type_info {
                IdlTypeVariant::Simple { r#type } => {
                    match r#type.as_str() {
                        "u8" => data.extend_from_slice(&(arg.as_str().unwrap().parse::<u8>().unwrap()).to_le_bytes()),
                        "u16" => data.extend_from_slice(&(arg.as_str().unwrap().parse::<u16>().unwrap()).to_le_bytes()),
                        "u32" => data.extend_from_slice(&(arg.as_str().unwrap().parse::<u32>().unwrap()).to_le_bytes()),
                        "u64" => data.extend_from_slice(&(arg.as_str().unwrap().parse::<u64>().unwrap()).to_le_bytes()),
                        "i8" => data.extend_from_slice(&(arg.as_str().unwrap().parse::<i8>().unwrap()).to_le_bytes()),
                        "i16" => data.extend_from_slice(&(arg.as_str().unwrap().parse::<i16>().unwrap()).to_le_bytes()),
                        "i32" => data.extend_from_slice(&(arg.as_str().unwrap().parse::<i32>().unwrap()).to_le_bytes()),
                        "i64" => data.extend_from_slice(&(arg.as_str().unwrap().parse::<i64>().unwrap()).to_le_bytes()),
                        "string" => {
                            let s = arg.as_str().unwrap();
                            data.extend_from_slice(&(s.len() as u32).to_le_bytes());
                            data.extend_from_slice(s.as_bytes());
                        },
                        _ => panic!("Unsupported type: {}", r#type)
                    }
                },
                _ => panic!("Complex types not yet supported")
            }
        }

        Instruction {
            program_id: Pubkey::from_str(&self.address).unwrap().to_string(),
            accounts,
            data,
        }
    }
}

// Example server setup
#[utoipa::path(
    post,
    path = "/upload_idl",
    request_body = AnchorIdl,
    responses(
        (status = 200, description = "IDL processed successfully", body = Vec<InstructionTemplate>)
    )   
)]
async fn upload_idl(
    Json(idl): Json<AnchorIdl>,
) -> Json<Vec<InstructionTemplate>> {
    let templates = AnchorIdl::process_idl(&serde_json::to_string(&idl).unwrap());
    Json(templates)
}

#[derive(Deserialize, ToSchema)]
struct PrepareInstructionPayload {
    accounts: Vec<AccountMeta>,
    args: Vec<Value>,
}

#[axum::debug_handler]
#[utoipa::path(
    post,
    path = "/prepare_instruction/{program_id}/{ix_name}",
    request_body = PrepareInstructionPayload,
    responses(
        (status = 200, description = "Instruction created successfully", body = Instruction)
    ),
    params(
        ("program_id" = String, Path, description = "Program ID"),
        ("ix_name" = String, Path, description = "Instruction name")
    )
)]
async fn prepare_instruction(
    Path((program_id, ix_name)): Path<(String, String)>,
    Json(payload): Json<PrepareInstructionPayload>,
) -> Json<Instruction> {
    let idl = AnchorIdl::load_cached(&program_id)
        .expect("IDL not found for program");
    
    let instruction = idl.generate_instruction(
        &ix_name,
        payload.accounts,
        payload.args
    );
    
    Json(instruction)
}

// Add these new  types with proper schema documentation
#[derive(Serialize, Deserialize, ToSchema)]
struct AccountMeta {
    pubkey: String,  // Base58 encoded public key
    is_signer: bool,
    is_writable: bool
}

#[derive(Serialize, Deserialize, ToSchema)]
struct Instruction {
    program_id: String,  // Base58 encoded program ID
    accounts: Vec<AccountMeta>,
    data: Vec<u8>
}

#[derive(Serialize, ToSchema)]
struct EnhancedInstructionRequirements {
    accounts: Vec<EnhancedAccountRequirement>,
    arguments: Vec<EnhancedArgumentRequirement>,
    known_addresses: HashMap<String, String>, // For system_program etc
}

#[derive(Serialize, ToSchema)]
struct EnhancedAccountRequirement {
    name: String,
    is_signer: bool,
    is_writable: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pda_derivation: Option<EnhancedPdaInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    known_address: Option<String>,
}

#[derive(Serialize, ToSchema)]
struct EnhancedPdaInfo {
    seeds: Vec<EnhancedSeedInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    program_id: Option<String>,
}

#[derive(Serialize, ToSchema)]
struct EnhancedSeedInfo {
    seed_type: String, // "constant", "account", etc
    #[serde(skip_serializing_if = "Option::is_none")]
    constant_value: Option<String>, // Base58 or UTF-8 string for readability
    #[serde(skip_serializing_if = "Option::is_none")]
    account_dependency: Option<String>,
}

#[derive(Serialize, ToSchema)]
struct EnhancedArgumentRequirement {
    name: String,
    type_info: String, // Simplified type description
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
}

#[utoipa::path(
    get,
    path = "/instruction_requirements/{program_id}/{ix_name}",
    responses(
        (status = 200, description = "Instruction requirements", body = EnhancedInstructionRequirements)
    )
)]
async fn get_instruction_requirements(
    Path((program_id, ix_name)): Path<(String, String)>,
) -> Json<EnhancedInstructionRequirements> {
    let idl = AnchorIdl::load_cached(&program_id)
        .expect("IDL not found");
    
    let ix = idl.instructions.iter()
        .find(|i| i.name == ix_name)
        .expect("Instruction not found");

    // Known program addresses
    let mut known_addresses = HashMap::new();
    known_addresses.insert(
        "system_program".to_string(), 
        "11111111111111111111111111111111".to_string()
    );
    known_addresses.insert(
        "token_program".to_string(),
        "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA".to_string()
    );
    known_addresses.insert(
        "associated_token_program".to_string(),
        "ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL".to_string()
    );

    let accounts = ix.accounts.iter().map(|acc| {
        let known_address = known_addresses.get(&acc.name).cloned();
        
        let pda_derivation = acc.pda.as_ref().map(|pda| {
            EnhancedPdaInfo {
                seeds: pda.seeds.iter().map(|seed| {
                    match seed.kind.as_str() {
                        "const" => {
                            let bytes = seed.value.get("value")
                                .and_then(|v| v.as_array())
                                .map(|arr| arr.iter()
                                    .filter_map(|v| v.as_u64())
                                    .map(|n| n as u8)
                                    .collect::<Vec<u8>>())
                                .unwrap_or_default();
                            
                            // Try to convert to UTF-8 if possible
                            EnhancedSeedInfo {
                                seed_type: "constant".to_string(),
                                constant_value: Some(
                                    String::from_utf8(bytes.clone())
                                        .unwrap_or_else(|_| bs58::encode(bytes).into_string())
                                ),
                                account_dependency: None,
                            }
                        },
                        "account" => EnhancedSeedInfo {
                            seed_type: "account".to_string(),
                            constant_value: None,
                            account_dependency: seed.value.get("path")
                                .and_then(|v| v.as_str())
                                .map(String::from),
                        },
                        _ => EnhancedSeedInfo {
                            seed_type: "unknown".to_string(),
                            constant_value: None,
                            account_dependency: None,
                        },
                    }
                }).collect(),
                program_id: pda.program.clone()
                    .map(|p| p.to_string())
                    .map(|s| s.replace("\"", "").replace("{", "").replace("}", "")),
            }
        });

        EnhancedAccountRequirement {
            name: acc.name.clone(),
            is_signer: acc.signer.unwrap_or(false),
            is_writable: acc.writable.unwrap_or(false),
            pda_derivation,
            known_address,
        }
    }).collect();

    let arguments = ix.args.iter().map(|arg| {
        EnhancedArgumentRequirement {
            name: arg.name.clone(),
            type_info: match &arg.type_info {
                IdlTypeVariant::Simple { r#type } => r#type.clone(),
                IdlTypeVariant::Detailed { type_info } => format!("{:?}", type_info),
                IdlTypeVariant::Defined { r#type } => format!("defined: {:?}", r#type),
            },
            description: None,
        }
    }).collect();

    Json(EnhancedInstructionRequirements { 
        accounts,
        arguments,
        known_addresses,
    })
}

#[derive(OpenApi)]
#[openapi(
    paths(
        upload_idl,
        prepare_instruction,
        get_program_instructions,
        get_instruction_requirements,
        prepare_transaction,
    ),
    components(
        schemas(
            InstructionTemplate,
            RequiredAccount,
            PdaDerivation,
            SeedComponent,
            ArgumentSpec,
            IdlTypeInfo,
            PrepareInstructionPayload,
            AnchorIdl,
            IdlMetadata,
            IdlInstruction,
            IdlAccount,
            IdlType,
            IdlField,
            IdlTypeVariant,
            IdlAccountItem,
            IdlPda,
            IdlSeed,
            IdlProgramSeed,
            AccountMeta,    // Add these new schemas
            Instruction,    // Add these new schemas
            EnhancedInstructionRequirements,
            EnhancedAccountRequirement,
            EnhancedPdaInfo,
            EnhancedSeedInfo,
            EnhancedArgumentRequirement,
            TransactionPayload,
            InstructionPayload,
            ComputeBudgetConfig,
            Transaction,
        )
    ),
    tags(
        (name = "Solana Program Interface", description = "Solana Program IDL Management API")
    )
)]
struct ApiDoc;

// Add new endpoint to get all instructions for a program
#[utoipa::path(
    get,
    path = "/program/{program_id}/instructions",
    responses(
        (status = 200, description = "List of all instructions", body = Vec<InstructionTemplate>)
    )
)]
async fn get_program_instructions(
    Path(program_id): Path<String>,
) -> Json<Vec<InstructionTemplate>> {
    let idl = AnchorIdl::load_cached(&program_id)
        .expect("IDL not found");
    
    Json(AnchorIdl::process_idl(&serde_json::to_string(&idl).unwrap()))
}

// Add these new types
#[derive(Deserialize, ToSchema)]
struct TransactionPayload {
    instructions: Vec<InstructionPayload>,
    compute_budget: Option<ComputeBudgetConfig>,
}

#[derive(Deserialize, ToSchema)]
struct InstructionPayload {
    program_id: String,
    instruction_name: String,
    accounts: Vec<AccountMeta>,
    args: Vec<Value>,
}

#[derive(Deserialize, ToSchema)]
struct ComputeBudgetConfig {
    units: Option<u32>,      // Compute unit limit
    price: Option<u64>,      // Compute unit price in micro-lamports
}

#[derive(Serialize, ToSchema)]
struct Transaction {
    instructions: Vec<Instruction>,
    message: String,
    blockhash: String,
    signatures_required: Vec<String>
}

// Add new endpoint
#[utoipa::path(
    post,
    path = "/prepare_transaction",
    request_body = TransactionPayload,
    responses(
        (status = 200, description = "Transaction created successfully", body = Transaction)
    )
)]
async fn prepare_transaction(
    Json(payload): Json<TransactionPayload>,
) -> Json<Transaction> {
    let mut instructions = Vec::new();
    
    // Add compute budget instructions if specified
    if let Some(budget) = payload.compute_budget {
        if let Some(units) = budget.units {
            instructions.push(create_compute_unit_limit_ix(units));
        }
        if let Some(price) = budget.price {
            instructions.push(create_compute_unit_price_ix(price));
        }
    }
    
    // Process each instruction
    for ix_payload in payload.instructions {
        let idl = AnchorIdl::load_cached(&ix_payload.program_id)
            .expect("IDL not found for program");
        
        let instruction = idl.generate_instruction(
            &ix_payload.instruction_name,
            ix_payload.accounts,
            ix_payload.args
        );
        
        instructions.push(instruction);
    }
    
    // Create a transaction message
    let message = solana_sdk::message::Message::new(
        &instructions.iter().map(|ix| {
            solana_sdk::instruction::Instruction {
                program_id: solana_sdk::pubkey::Pubkey::from_str(&ix.program_id).unwrap(),
                accounts: ix.accounts.iter().map(|meta| {
                    solana_sdk::instruction::AccountMeta {
                        pubkey: solana_sdk::pubkey::Pubkey::from_str(&meta.pubkey).unwrap(),
                        is_signer: meta.is_signer,
                        is_writable: meta.is_writable,
                    }
                }).collect(),
                data: ix.data.clone(),
            }
        }).collect::<Vec<_>>(),
        None
    );

    // Serialize and base64 encode the message
    let serialized = bincode::serialize(&message).unwrap();
    let encoded = base64::encode(&serialized);

    Json(Transaction { 
        instructions,
        message: encoded,
        blockhash: "".to_string(),
        signatures_required: vec![],
    })
}

// Helper functions for compute budget instructions
fn create_compute_unit_limit_ix(units: u32) -> Instruction {
    Instruction {
        program_id: "ComputeBudget111111111111111111111111111111".to_string(),
        accounts: vec![],
        data: {
            let mut data = vec![0x02]; // Instruction index for SetComputeUnitLimit
            data.extend_from_slice(&units.to_le_bytes());
            data
        },
    }
}

fn create_compute_unit_price_ix(price: u64) -> Instruction {
    Instruction {
        program_id: "ComputeBudget111111111111111111111111111111".to_string(),
        accounts: vec![],
        data: {
            let mut data = vec![0x03]; // Instruction index for SetComputeUnitPrice
            data.extend_from_slice(&price.to_le_bytes());
            data
        },
    }
}
// Fake IDL for Token program - we'll inject this into cache on startup
fn get_token_program_idl() -> AnchorIdl {
    AnchorIdl {
        address: "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA".to_string(),
        metadata: IdlMetadata {
            name: "token_program".to_string(),
            version: "3.0.0".to_string(), 
            spec: "1.0.0".to_string(),
            description: Some("Token Program".to_string()),
        },
        instructions: vec![

        IdlInstruction {
            name: "closeAccount".to_string(),
            discriminator: [9, 0, 0, 0, 0, 0, 0, 0], // Token program close_account instruction index
            accounts: vec![
                IdlAccountItem {
                    name: "account".to_string(),
                    writable: Some(true),
                    signer: Some(false),
                    pda: None,
                },
                IdlAccountItem {
                    name: "destination".to_string(),
                    writable: Some(true),
                    signer: Some(false),
                    pda: None,
                },
                IdlAccountItem {
                    name: "authority".to_string(),
                    writable: Some(false),
                    signer: Some(true),
                    pda: None,
                },
            ],
            args: vec![],
        },
            IdlInstruction {
                name: "transfer".to_string(),
                discriminator: [3, 0, 0, 0, 0, 0, 0, 0], // Token program transfer instruction index
                accounts: vec![
                    IdlAccountItem {
                        name: "source".to_string(),
                        writable: Some(true),
                        signer: Some(false),
                        pda: None,
                    },
                    IdlAccountItem {
                        name: "destination".to_string(), 
                        writable: Some(true),
                        signer: Some(false),
                        pda: None,
                    },
                    IdlAccountItem {
                        name: "authority".to_string(),
                        writable: Some(false),
                        signer: Some(true),
                        pda: None,
                    },
                ],
                args: vec![
                    IdlField {
                        name: "amount".to_string(),
                        type_info: IdlTypeVariant::Simple {
                            r#type: "u64".to_string()
                        }
                    }
                ],
            },
            IdlInstruction {
                name: "mint_to".to_string(),
                discriminator: [7, 0, 0, 0, 0, 0, 0, 0], // Token program mint_to instruction index
                accounts: vec![
                    IdlAccountItem {
                        name: "mint".to_string(),
                        writable: Some(true),
                        signer: Some(false),
                        pda: None,
                    },
                    IdlAccountItem {
                        name: "account".to_string(),
                        writable: Some(true),
                        signer: Some(false),
                        pda: None,
                    },
                    IdlAccountItem {
                        name: "authority".to_string(),
                        writable: Some(false),
                        signer: Some(true),
                        pda: None,
                    },
                ],
                args: vec![
                    IdlField {
                        name: "amount".to_string(),
                        type_info: IdlTypeVariant::Simple {
                            r#type: "u64".to_string()
                        }
                    }
                ],
            },
            IdlInstruction {
                name: "burn".to_string(),
                discriminator: [8, 0, 0, 0, 0, 0, 0, 0], // Token program burn instruction index
                accounts: vec![
                    IdlAccountItem {
                        name: "account".to_string(),
                        writable: Some(true),
                        signer: Some(false),
                        pda: None,
                    },
                    IdlAccountItem {
                        name: "mint".to_string(),
                        writable: Some(true),
                        signer: Some(false),
                        pda: None,
                    },
                    IdlAccountItem {
                        name: "authority".to_string(),
                        writable: Some(false),
                        signer: Some(true),
                        pda: None,
                    },
                ],
                args: vec![
                    IdlField {
                        name: "amount".to_string(),
                        type_info: IdlTypeVariant::Simple {
                            r#type: "u64".to_string()
                        }
                    }
                ],
            }
        ],
        accounts: vec![],
        types: vec![]
    }
}


// Fake IDL for ATA program - we'll inject this into cache on startup
fn get_ata_idl() -> AnchorIdl {
    AnchorIdl {
        address: "ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL".to_string(),
        metadata: IdlMetadata {
            name: "associated_token".to_string(),
            version: "1.0.0".to_string(),
            spec: "1.0.0".to_string(),
            description: Some("Associated Token Account Program".to_string()),
        },
        instructions: vec![
            IdlInstruction {
                name: "create".to_string(),
                discriminator: [0xd1, 0x03, 0x59, 0x64, 0x39, 0x5f, 0x6c, 0x4c], // Real ATA discriminator
                accounts: vec![
                    IdlAccountItem {
                        name: "payer".to_string(),
                        writable: Some(true),
                        signer: Some(true),
                        pda: None,
                    },
                    IdlAccountItem {
                        name: "associated_token".to_string(),
                        writable: Some(true),
                        signer: Some(false),
                        pda: Some(IdlPda {
                            seeds: vec![
                                IdlSeed {
                                    kind: "account".to_string(),
                                    value: serde_json::json!({
                                        "path": "wallet"
                                    }),
                                },
                                IdlSeed {
                                    kind: "const".to_string(),
                                    value: serde_json::json!({
                                        "value": [6, 221, 246, 225, 215, 101, 161, 147, 217, 203, 225, 70, 206, 235, 121, 172, 28, 180, 133, 237, 95, 91, 55, 145, 58, 140, 245, 133, 126, 255, 0, 169]
                                    }),
                                },
                                IdlSeed {
                                    kind: "account".to_string(),
                                    value: serde_json::json!({
                                        "path": "mint"
                                    }),
                                },
                            ],
                            program: Some(IdlProgramSeed {
                                kind: "const".to_string(),
                                value: serde_json::json!("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"),
                            }),
                        }),
                    },
                    IdlAccountItem {
                        name: "wallet".to_string(),
                        writable: Some(false),
                        signer: Some(false),
                        pda: None,
                    },
                    IdlAccountItem {
                        name: "mint".to_string(),
                        writable: Some(false),
                        signer: Some(false),
                        pda: None,
                    },
                    IdlAccountItem {
                        name: "system_program".to_string(),
                        writable: Some(false),
                        signer: Some(false),
                        pda: None,
                    },
                    IdlAccountItem {
                        name: "token_program".to_string(),
                        writable: Some(false),
                        signer: Some(false),
                        pda: None,
                    },
                ],
                args: vec![],
            },
        ],
        accounts: vec![],
        types: vec![],
    }
}

// Helper function to create ATA instruction
pub fn create_ata_ix(
    payer: &str,
    wallet: &str,
    mint: &str,
) -> Instruction {
    Instruction {
        program_id: "ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL".to_string(),
        accounts: vec![
            AccountMeta {
                pubkey: payer.to_string(),
                is_signer: true,
                is_writable: true,
            },
            AccountMeta {
                pubkey: get_ata_address(wallet, mint),
                is_signer: false,
                is_writable: true,
            },
            AccountMeta {
                pubkey: wallet.to_string(),
                is_signer: false,
                is_writable: false,
            },
            AccountMeta {
                pubkey: mint.to_string(),
                is_signer: false,
                is_writable: false,
            },
            AccountMeta {
                pubkey: "11111111111111111111111111111111".to_string(),
                is_signer: false,
                is_writable: false,
            },
            AccountMeta {
                pubkey: "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA".to_string(),
                is_signer: false,
                is_writable: false,
            },
        ],
        data: vec![0xd1, 0x03, 0x59, 0x64, 0x39, 0x5f, 0x6c, 0x4c], // ATA create instruction discriminator
    }
}

// Helper function to derive ATA address
pub fn get_ata_address(wallet: &str, mint: &str) -> String {
    spl_associated_token_account::get_associated_token_address(
        &Pubkey::from_str(wallet).unwrap(),
        &Pubkey::from_str(mint).unwrap()
    ).to_string()
}

// Update main() to inject ATA IDL into cache on startup
#[tokio::main]
async fn main() {
    // Inject ATA IDL into cache
    let ata_idl = get_ata_idl();
    CACHE.lock().unwrap().insert(
        "ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL".to_string(),
        ata_idl
    );
    let token_idl = get_token_program_idl();
    CACHE.lock().unwrap().insert(
        "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA".to_string(),
        token_idl
    );
    let app = Router::new()
        .route("/upload_idl", post(upload_idl))
        .route("/prepare_instruction/:program_id/:ix_name", post(prepare_instruction))
        .route("/program/:program_id/instructions", get(get_program_instructions))
        .route("/instruction_requirements/:program_id/:ix_name", 
               get(get_instruction_requirements))
        .route("/prepare_transaction", post(prepare_transaction))
        // Add Swagger UI
        .merge(SwaggerUi::new("/swagger-ui")
            .url("/api-docs/openapi.json", ApiDoc::openapi()));

    axum_server::bind(SocketAddr::from(([0, 0, 0, 0], 3000)))
        .serve(app.into_make_service())
        .await
        .unwrap();
}